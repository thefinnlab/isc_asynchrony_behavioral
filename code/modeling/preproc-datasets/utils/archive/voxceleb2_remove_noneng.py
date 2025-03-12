import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob
import itertools
import math 

import shutil
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from natsort import natsorted

from pathlib import Path

sys.path.append('../../utils/')

import utils 

def scan_files_for_id(path, diff_id):
    return glob.glob(os.path.join(path, f"{diff_id}*"))

def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def move_file(fn, dst_dir):

    # fns = list(Path(src_dir).glob(f'{vid_id}*'))
    # # fns = glob.glob(os.path.join(src_dir, f'{vid_id}*'))

    # for fn in fns:

    dst_path = os.path.join(dst_dir, os.path.basename(fn))

    if os.path.exists(fn):
        try:
            shutil.move(fn, dst_path)
            return True, f"Moved {fn} to {dst_path}"
        except Exception as e:
            return False, f"Error moving {fn}: {str(e)}"
    elif os.path.exists(dst_path):
        return True, f"Moved clips from {fn} to {dst_path}"
    else:
        return False, f"Source directory not found: {fn}"

def reorganize_files(fns, dst_dir, num_jobs=None):
    """
    Reorganize files for multiple data splits using parallel processing.
    
    Args:
        base_dir (str): Base directory containing data splits
        splits (list): List of data splits (e.g., ['train', 'val', 'test'])
        targets (dict): Mapping of key prefixes to target directories
        max_workers (int, optional): Number of parallel workers. Defaults to None (auto).
    """

    # Use 75% of available cores by default, but at least 1
    if not num_jobs:
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    print(f"Found {len(fns)} different ids")

    to_process_count = len(fns)
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit jobs
        future_to_file = {
            executor.submit(move_file, fn, dst_dir): fn 
            for fn in fns
        }

        # Process results as they complete
        pbar = tqdm(total=to_process_count, desc=f"Removing non-eng fiels")
        
        # Collect results
        success = 0
        error = 0
        for future in as_completed(future_to_file):
            fn = future_to_file[future]
            try:
                success, msg = future.result()
                success += 1
                # print(f"Processed {json_file}: Moved {len(moved_files)} files, Updated: {paths_updated}")
            except Exception as e:
                error += 1
                print(f"Error processing {fn}: {e}", flush=True)

            pbar.update(1)

    print(f" Files moved: {success}")
    print(f" Failed: {error}")

def main():

    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')
    parser.add_argument('-d','--dataset', type=str,required=True, 
                      help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--split', type=str, default=None,
                      help='Which split to process')

    ### Model names
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2', 
                        help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default=None,
                        help='Video model to use, or None to skip video processing')

    parser.add_argument('--num_shards', type=int, default=1,
                    help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0,
                    help='Current shard to process (0-based indexing)')
    parser.add_argument('--num_jobs', type=int, default=1,
                    help='Number of jobs for multiprocessing')

    ### Sharding for more efficient processing
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Force extraction even if files exist')

    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val', 'test']
    
    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    # Prepare directory structure --> only this script is needed for video
    dirs, splits = utils.prepare_directory_structure(
        output_dir, 
        splits, 
        dir_names=['audio', 'textgrids', 'video', 'prosody', 'transcripts'],
    )

    ###########################################
    ########## Load the ID files ##############
    ###########################################

    eng_ids_fn = os.path.join(output_dir, 'src', 'vox-en.id')

    with open(eng_ids_fn, 'r') as f:
        eng_ids = [line.strip() for line in f.readlines()]

    train_val_ids = [x for x in eng_ids if 'dev' in x]
    test_ids = [x for x in eng_ids if 'test' in x]

    # join to our filenaming convention and find unique ids
    train_val_ids = set([
        '_'.join(x.strip('dev/mp4/').split('/')[:-1])
        for x in train_val_ids
    ])

    test_ids = set([
        '_'.join(x.strip('test/mp4/').split('/')[:-1])
        for x in test_ids
    ])

    train_val_ids = natsorted(train_val_ids)
    test_ids = natsorted(test_ids)

    ###########################################
    ########## Create model dirs ##############
    ###########################################

    model_cache_dirs = {
        'prosody_model': 'prosody',
        'text_model': args.text_model,
    }

    # Set name of the model combo for the metadata
    model_combo = args.text_model

    if args.audio_model:
        model_cache_dirs['audio_model'] = args.audio_model
        model_combo += f'-{args.audio_model}'

    if args.video_model:
        model_cache_dirs['video_model'] = args.video_model
        model_combo += f'-{args.video_model}'

    # Create cache for our features and a temp directory for writing progress
    dirs["cache_dir"] = os.path.join(args.output_dir, 'features')
    dirs["metadata_dir"] = os.path.join(dirs["cache_dir"], 'metadata', model_combo)

    for model_type, dir_name in model_cache_dirs.items():
        model_cache_dir = os.path.join(dirs["cache_dir"], dir_name)
        os.makedirs(model_cache_dir, exist_ok=True)

        # Add to the directories
        dirs[model_type] = model_cache_dir

    ###########################################
    ######## Go through each split ############
    ###########################################

    num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    for split in splits:

        print(f"\nChecking features for {split} split...", flush=True)
        split_dirs = {k: os.path.join(v, split) for k, v in dirs.items()}

        # Make the json directories
        split_dirs["temp_dir"] = os.path.join(split_dirs["metadata_dir"], 'temp')
        split_dirs["errors_dir"] = os.path.join(split_dirs["metadata_dir"], 'errors')

        if split == 'test':
            split_ids = test_ids
        else:
            split_ids = train_val_ids

        # Source the difference ids from the original directory (video)
        dir_fns = glob.glob(os.path.join(split_dirs['video'], '*'))

        # Find all the speaker IDs
        dir_file_ids = set([
            '_'.join(os.path.basename(fn).split('_')[:-1])
            for fn in dir_fns
        ])

        # Find set difference between all ids present in the directory and the split_ids
        difference_ids = set(dir_file_ids).difference(split_ids)

        print (f"Found total {len(difference_ids)} / {len(dir_file_ids)} non-english IDs for {split}", flush=True)

        difference_ids = list(difference_ids)

        # Apply sharding logic
        if args.num_shards > 1:
            # Calculate shard size and starting/ending indices
            shard_size = math.ceil(len(difference_ids) / args.num_shards)
            start_idx = args.current_shard * shard_size
            end_idx = min(start_idx + shard_size, len(difference_ids))
            
            # Get only the files for the current shard
            difference_ids = difference_ids[start_idx:end_idx]
            
            print(f"Processing shard {args.current_shard+1}/{args.num_shards} with {len(difference_ids)} IDS", flush=True)

        # Go through each directory within the current split
        # for dtype, path in split_dirs.items():

        for dtype in ['temp_dir', 'errors_dir']:

            print (f"Running {dtype}", flush=True)

            path = split_dirs[dtype]

            # print (f"Found total {len(difference_fns)} for {dtype}", flush=True)
            # Create a list of arguments to pass to scan_files_for_id
            tasks = [(path, diff_id) for diff_id in difference_ids]

            # Process results as they complete
            pbar = tqdm(total=len(tasks), desc=f"Scanning dirs for non-eng ids")
            difference_fns = []

            # Use ThreadPoolExecutor to run the scans in parallel
            with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:

                # Start scanning with the progress bar
                futures = {executor.submit(scan_files_for_id, path, diff_id): diff_id for path, diff_id in tasks}
                
                # Initialize the tqdm progress bar
                for future in as_completed(futures):
                    result = future.result()  # Get the result (list of filenames)
                    difference_fns.extend(result)  # Add the result to the list of filenames
                    pbar.update(1)  # Update progress bar as each task completes

            dst_dir = os.path.join(path, 'non-eng')
            ensure_directory_exists(dst_dir)

            # # Go through each test_id
            reorganize_files(fns=difference_fns, dst_dir=dst_dir, num_jobs=args.num_jobs)

if __name__ == '__main__':
    main()