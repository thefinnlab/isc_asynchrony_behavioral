import os, sys
import glob
import json
import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc  # Added for garbage collection

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/utils/')

from config import *

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/preproc-datasets/utils/')

import utils

def strip_fn(file_name, dir_type, dir_exts):
    # Clean up filename based on dir_type
    if dir_type == 'text_model':
        file_name = file_name.strip('_attention-mask').strip('_text-tokens')
    elif dir_type == 'prosody_model':
        file_name = file_name.strip('_prominence').strip('_boundary')
    elif dir_type == 'audio_model':
        file_name = file_name.strip('_audio-features')
    elif dir_type == 'video_model':
        file_name = file_name.strip('_video-features')
    elif dir_type == 'temp_dir' or dir_type == 'errors_dir':
        file_name = file_name.strip('_processed')

    return file_name


def process_file(fn, split_lang_files, dst_dir, dir_type, dir_exts):
    """Process a single file: move or remove based on criteria"""
    basename = os.path.basename(fn)
    file_name = os.path.splitext(basename)[0]
    file_name = strip_fn(file_name, dir_type, dir_exts)

    # Check if file is valid
    try:
        if file_name in split_lang_files:
            dst_path = os.path.join(dst_dir, basename)
            shutil.move(fn, dst_path)
            return True, fn, "Success"
        else:
            # os.remove(fn)
            return False, fn, "Invalid file"
    except Exception as e:
        return False, fn, str(e)

def sort_files_between_splits(args, split_dfs, src_split, dst_split, src_dir, dst_dir, dir_type, dir_exts):
    '''
    We check the src_split directory for dst_split files
    '''
    print (f"Moving files from {src_split} to {dst_split}")

    # Grab the information from destination split (the one which we're looking to move files to)
    dst_df = split_dfs[dst_split]

    # Grab dst split filenames
    dst_lang_df = dst_df[dst_df['most_common_lang'] == lang_id]
    dst_lang_files = dst_lang_df['clip_id'].apply(lambda x: os.path.splitext(x)[0]).tolist()
    # dst_lang_files = dst_lang_df['composite_key'].apply(lambda x: os.path.splitext(x)[0]).tolist()
    dst_set = set(dst_lang_files)

    # Find all src split files
    src_fns = glob.glob(os.path.join(src_dir, f'*.{dir_exts[dir_type]}'))
    src_fns = utils.get_shard_data(src_fns, num_shards=args.num_shards, current_shard=args.current_shard)

    print(f"Processing {len(src_fns)} files for {dir_type}...", flush=True)

    # Create a mapping from stripped filenames to full paths
    src_filename_to_path = {}
    for fn in src_fns:
        file_name = os.path.splitext(os.path.basename(fn))[0]
        file_name = strip_fn(file_name, dir_type, dir_exts)
        src_filename_to_path[file_name] = fn

    # Find overlapping files (filenames present in both src and dst)
    overlapping_filenames = dst_set.intersection(src_filename_to_path.keys())
    
    # Get the full paths of the overlapping files
    src_fns = [src_filename_to_path[filename] for filename in overlapping_filenames]

    # Set up a process pool and run with progress bar
    completed = 0
    invalid_files = 0
    errors = 0

    # Adjust chunk size for better performance
    chunk_size = min(max(1, len(src_fns) // (args.num_jobs * 4)), 100)
    
    with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
        # Submit all tasks with improved chunking
        future_to_job = {
            # Check the src directory for dst lang files
            executor.submit(process_file, fn, dst_lang_files, dst_dir, dir_type, dir_exts): fn
            for fn in src_fns
        }

        # Process results as they complete
        pbar = tqdm(total=len(src_fns), desc=f"Moving files from {src_split} to {dst_split}")

        for future in as_completed(future_to_job):
            file_path = future_to_job[future]

            try:
                success, _, message = future.result()
                if success:
                    completed += 1
                else:
                    invalid_files += 1
            except Exception as e:
                errors += 1
    
            pbar.update(1)
            
            # Periodically force garbage collection
            if (completed + invalid_files + errors) % 1000 == 0:
                gc.collect()

        pbar.close()

    print(f"Total files moved from {src_split} to {dst_split}: {completed}")
    print(f"Total invalid files: {invalid_files}")
    print(f"Total errors: {errors}")

def sort_files_for_split(args, df_split, lang_id, dir_path, dir_type, dir_exts):

    # Add explicit garbage collection between splits
    gc.collect()

    print (f"Cleaning and organizing {split}, dir: {dir_type}")
    lang_dir = os.path.join(dir_path, lang_id)
    os.makedirs(lang_dir, exist_ok=True)

    df_lang_split = df_split[df_split['most_common_lang'] == lang_id]
    split_lang_files = df_lang_split['clip_id'].apply(lambda x: os.path.splitext(x)[0]).tolist()
    # split_lang_files = df_lang_split['composite_key'].apply(lambda x: os.path.splitext(x)[0]).tolist()
    split_set = set(split_lang_files)

    print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)

    all_fns = glob.glob(os.path.join(dir_path, f'*.{dir_exts[dir_type]}'))

    # Create a mapping from stripped filenames to full paths
    filename_to_path = {}
    for fn in all_fns:
        file_name = os.path.splitext(os.path.basename(fn))[0]
        file_name = strip_fn(file_name, dir_type, dir_exts)
        filename_to_path[file_name] = fn

    # Find overlapping files (filenames present in both src and dst)
    overlapping_filenames = split_set.intersection(filename_to_path.keys())

    # Get the full paths of the overlapping files
    all_fns = [filename_to_path[filename] for filename in overlapping_filenames]

    # Skip empty directories
    if not all_fns:
        print(f"No files found for {dir_type}, skipping...", flush=True)
        return None

    # Apply sharding logic --> divide dataset into number of shards 
    all_fns = utils.get_shard_data(all_fns, num_shards=args.num_shards, current_shard=args.current_shard)

    print(f"Processing {len(all_fns)} files for {dir_type}...", flush=True)
    
    # Set up a process pool and run with progress bar
    completed = 0
    invalid_files = 0
    errors = 0

    # Adjust chunk size for better performance
    chunk_size = min(max(1, len(all_fns) // (args.num_jobs * 4)), 100)
    
    with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
        # Submit all tasks with improved chunking
        future_to_job = {
            executor.submit(process_file, fn, split_lang_files, lang_dir, dir_type, dir_exts): fn
            for fn in all_fns
        }

        # Process results as they complete
        pbar = tqdm(total=len(all_fns), desc=f"Extracting {dir_type} from {split}")

        for future in as_completed(future_to_job):
            file_path = future_to_job[future]

            try:
                success, _, message = future.result()
                if success:
                    completed += 1
                else:
                    invalid_files += 1
            except Exception as e:
                errors += 1
    
            pbar.update(1)
            
            # Periodically force garbage collection
            if (completed + invalid_files + errors) % 1000 == 0:
                gc.collect()

        pbar.close()

        print(f"Total files moved: {completed} / {len(split_lang_files)}")
        print(f"Total invalid files: {invalid_files}")
        print(f"Total errors: {errors}")
        
        # Clear references to reduce memory usage
        all_fns = None
        executor = None
        future_to_job = None
        pbar = None
        gc.collect()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')

    ### Model names
    parser.add_argument('--dataset', type=str, default='voxceleb2', help='Dataset for processing')
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2',
                    help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default='data2vec',
                    help='Video model to use, or None to skip video processing')

    parser.add_argument('--num_jobs', type=int, default=1,
                    help='Number of shards to divide the dataset into')

    ### Sharding for more efficient processing
    parser.add_argument('--num_shards', type=int, default=1,
                    help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0,
                    help='Current shard to process (0-based indexing)')
    parser.add_argument('--overwrite', type=int, default=0,
                    help='Force extraction even if files exist')

    args = parser.parse_args()

    dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', args.dataset)

    # Grab the source directories + make the new splits
    lang_id = 'eng'
    splits = ['train', 'val', 'test']
    dir_names = ['audio', 'prosody', 'transcripts', 'textgrids']

    dir_exts = {
        # 'temp_dir': 'json',
        # 'errors_dir': 'json',
        'audio': 'wav',
        'prosody': 'prom',
        'transcripts': 'txt',
        'textgrids': 'TextGrid',
        # 'prosody_model': 'pt',
        # 'text_model': 'pt',
        # 'audio_model': 'pt',
        # 'video_model': 'pt',
    }

    dirs, _ = utils.prepare_directory_structure(
        dataset_dir, 
        dir_names=dir_names
    )

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

    # Add model cache dirs
    for model_type, dir_name in model_cache_dirs.items():
        model_cache_dir = os.path.join(dataset_dir, 'features', dir_name)
        os.makedirs(model_cache_dir, exist_ok=True)

        # Add to the directories
        dirs[model_type] = model_cache_dir

    # Load info for each split
    #  'src', split,  # ADd for avspeech
    if args.dataset == 'avspeech':
        split_dfs = {split: pd.read_csv(os.path.join(dataset_dir, 'src', split, f"{split}_metadata-filtered.csv")) for split in splits}    
    else:
        split_dfs = {split: pd.read_csv(os.path.join(dataset_dir, f"{split}_metadata-filtered.csv")) for split in splits} 
        
    swap_splits = [
        ('train', 'val'),
        ('val', 'train')
    ]

    for dir_type, dir_ext in dir_exts.items():

        if dir_type not in ['temp_dir', 'errors_dir']:
            dir_path = dirs[dir_type]

        # Sort the files between the directories
        for src_split, dst_split in swap_splits:

            # Set the src and dst directories
            if dir_type in ['temp_dir', 'errors_dir']:
                src_dir = os.path.join(dataset_dir, 'features', 'metadata', model_combo, src_split, dir_type.replace('_dir', '')) # lang_id
                dst_dir = os.path.join(dataset_dir, 'features', 'metadata', model_combo, dst_split, dir_type.replace('_dir', '')) # lang_id
            else:
                src_dir = os.path.join(dir_path, src_split, 'eng')
                dst_dir = os.path.join(dir_path, dst_split, 'eng')

            sort_files_between_splits(args, split_dfs, src_split, dst_split, src_dir, dst_dir, dir_type, dir_exts)

        # Now go through each split and sort english / not english
        for split in splits:

            print(f"\nFinding files for {split} split...", flush=True)

            if dir_type in ['temp_dir', 'errors_dir']:
                split_dir = os.path.join(dataset_dir, 'features', 'metadata', model_combo, split, dir_type.replace('_dir', ''))
            else:
                split_dir = os.path.join(dir_path, split)

            sort_files_for_split(args, split_dfs[split], lang_id, split_dir, dir_type, dir_exts)