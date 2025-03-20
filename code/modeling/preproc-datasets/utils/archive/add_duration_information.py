import os
import sys
import glob
import argparse
import json
import shutil
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/utils/')

from config import *

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/preproc-datasets/utils/')

import utils

def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def add_duration_information(audio_dir, split, fn):
    data = utils.load_json(fn)

    if 'duration' in data:
        return True
    # try:
    audio_fn = os.path.join(audio_dir, f"{data['base_name']}.wav")

    if not os.path.exists(audio_fn):
        split_basename = data['base_name'].split('_')
        speaker_id = split_basename[0]
        video_id = '_'.join(split_basename[1:-1])
        clip_id = split_basename[-1]

        if split in ['train', 'val']:
            speaker_dir = audio_dir.replace(f'audio/{split}', 'src/dev')
        else:
            speaker_dir = audio_dir.replace(f'audio/{split}', 'src/test')

        audio_fn = os.path.join(speaker_dir, speaker_id, video_id, f"{clip_id}.mp4")

    waveform, sr = utils.load_audio(audio_fn)
    duration = waveform.shape[-1] / sr
    data['duration'] = duration

    utils.save_json(fn, data)

def main():
    #
    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')

    ### Model names
    parser.add_argument('--dataset', type=str, default='voxceleb2', help='Dataset for processing')
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
    lang_id = 'eng'

    # Splits to process
    splits = ['train', 'val', 'test']

    for split in splits:

        if args.dataset == 'voxceleb2':
            metadata_dir = os.path.join(dataset_dir, 'features', 'metadata', 'gpt2-wav2vec2-data2vec', split, 'temp', lang_id)
            audio_dir = os.path.join(dataset_dir, 'audio', split, lang_id)
        elif args.dataset == 'avspeech':
            metadata_dir = os.path.join(dataset_dir, 'features', 'metadata', 'gpt2-wav2vec2-data2vec', split, lang_id, 'temp')
            audio_dir = os.path.join(dataset_dir, 'audio', split, lang_id)
        else:
            metadata_dir = os.path.join(dataset_dir, 'features', 'metadata', 'gpt2-wav2vec2-data2vec', split, 'temp')
            audio_dir = os.path.join(dataset_dir, 'audio', split)
        
        metadata_fns = glob.glob(os.path.join(metadata_dir, '*.json'))
        metadata_fns = utils.get_shard_data(metadata_fns, num_shards=args.num_shards, current_shard=args.current_shard)

        print(f"\nProcessing {split} split:")
        print(f"Found {len(metadata_fns)} JSON files")

        to_process_count = len(metadata_fns)
    
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
            # Submit jobs
            future_to_file = {
                executor.submit(add_duration_information, audio_dir, split, fn): fn 
                for fn in metadata_fns
            }

            # Process results as they complete
            pbar = tqdm(total=to_process_count, desc=f"Extracting audio from {split} videos")
            
            # Collect results
            succeeded = 0
            failed = 0

            for future in as_completed(future_to_file):
                fn = future_to_file[future]
                try:
                    success = future.result()
                    succeeded += 1
                except Exception as e:
                    failed += 1

                pbar.update(1)

        print(f"{split.upper()} split summary:")
        print(f"Succeed: {succeeded}")
        print(f"Failed: {failed}")

if __name__ == '__main__':
    main()