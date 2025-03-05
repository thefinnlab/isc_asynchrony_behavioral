import os
import sys
import json
import shutil
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def process_json_file(json_file_path: str, base_input_dir: str, targets: Dict[str, str]) -> Tuple[List[str], bool]:
    """
    Process a single JSON file:
    1. Move associated files to target directories
    2. Update paths in the JSON file
    
    Args:
        json_file_path (str): Path to the JSON file
        base_input_dir (str): Base input directory
        targets (dict): Mapping of key prefixes to target directories
    
    Returns:
        Tuple of (moved files list, whether file was updated)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Track moved files and whether paths were updated
        moved_files = []
        paths_updated = False
        
        # Create a copy of the data to modify
        updated_data = data.copy()
        
        # Process each path in the JSON
        for key, path in data.items():
            # Check if the key matches any target directories
            for prefix, target_dir in targets.items():
                if key.startswith(prefix):
                    # Ensure target directory exists
                    ensure_directory_exists(target_dir)
                    
                    # Get the filename from the original path
                    filename = os.path.basename(path)
                    
                    # Construct full source and destination paths
                    full_source_path = os.path.join(base_input_dir, filename)
                    full_dest_path = os.path.join(target_dir, filename)
                    
                    # Move the file
                    if os.path.exists(full_source_path):
                        shutil.move(full_source_path, full_dest_path)
                        moved_files.append(full_source_path)
                        
                        # Update the path in the JSON data
                        updated_data[key] = full_dest_path
                        paths_updated = True
        
        # If paths were updated, write back to the JSON file
        if paths_updated:
            with open(json_file_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
        
        return moved_files, paths_updated
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return [], False

def reorganize_files(base_dir: str, split: str, targets: Dict[str, str], num_jobs: int = None):
    """
    Reorganize files for multiple data splits using parallel processing.
    
    Args:
        base_dir (str): Base directory containing data splits
        splits (list): List of data splits (e.g., ['train', 'val', 'test'])
        targets (dict): Mapping of key prefixes to target directories
        max_workers (int, optional): Number of parallel workers. Defaults to None (auto).
    """

    if num_jobs is None:
        # Use 75% of available cores by default, but at least 1
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    # Aggregate results across splits
    total_moved = 0
    total_updated = 0

    # Construct input directory for the current split
    input_dir = os.path.join(base_dir, split)
    
    # Find all JSON files in the current split
    json_files = [
        os.path.join(input_dir, 'temp', f) 
        for f in os.listdir(os.path.join(input_dir, 'temp')) 
        if f.endswith('.json')
    ]
    
    print(f"\nProcessing {split} split:")
    print(f"Input directory: {input_dir}")
    print(f"Found {len(json_files)} JSON files")

    to_process_count = len(json_files)
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit jobs
        future_to_file = {
            executor.submit(process_json_file, json_file, input_dir, targets): json_file 
            for json_file in json_files
        }

        # Process results as they complete
        pbar = tqdm(total=to_process_count, desc=f"Extracting audio from {split} videos")
        
        # Collect results
        split_moved = 0
        split_updated = 0
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                moved_files, paths_updated = future.result()
                split_moved += len(moved_files)
                split_updated += 1 if paths_updated else 0
                # print(f"Processed {json_file}: Moved {len(moved_files)} files, Updated: {paths_updated}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

            pbar.update(1)
        
        # Update overall totals
        total_moved += split_moved
        total_updated += split_updated
        
        print(f"{split.upper()} split summary:")
        print(f"  Files moved: {split_moved}")
        print(f"  JSON files updated: {split_updated}")
    
    # Final summary
    print("\nTotal reorganization summary:")
    print(f"Total files moved: {total_moved}")
    print(f"Total JSON files updated: {total_updated}")

def main():
    # Base directory for the dataset
    base_dir = '/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/gpt2-wav2vec2-data2vec/'
    
    # Splits to process
    splits = ['train', 'val', 'test']

    for split in splits:
        # Target directories configuration
        targets = {
            'text_tokens_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/gpt2/{split}/',
            'attention_mask_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/gpt2/{split}/',
            'prominence_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/prosody/{split}/',
            'boundary_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/prosody/{split}/',
            'audio_features_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/wav2vec2/{split}/',
            'video_features_path': f'/dartfs/rc/lab/F/FinnLab/datasets/nlp-datasets/voxceleb2/features/data2vec/{split}/'
        }
        
        # Run the reorganization
        reorganize_files(base_dir, split, targets)

if __name__ == '__main__':
    main()