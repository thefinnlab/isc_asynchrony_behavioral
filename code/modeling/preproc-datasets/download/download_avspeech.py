import os
import sys
import glob
import random
import shutil
import tarfile
import pandas as pd
import concurrent.futures
from typing import Dict, List, Tuple

# Assuming these imports work in your environment
sys.path.append('../../../utils/')
from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../')
import utils

# Constants
VALIDATION_PERCENTAGE = 0.1
RANDOM_SEED = 42


def untar_file(fn: str) -> bool:
    """
    Extract a tar file to a proper directory.
    
    Args:
        fn: Path to the tar file
        
    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    print(f"Extracting {fn}...")
    try:
        # Extract to parent directory instead of the tar file itself
        extract_dir = os.path.dirname(file_path)

        # Create the extraction directory if it doesn't exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
        with tarfile.open(fn, 'r') as tar:
            tar.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Error extracting {fn}: {e}")
        return False


def process_tar_file(fn: str, split_ids: Dict[str, List[str]], 
                    dirs: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Process a single tar file and move its contents to the appropriate split directories.
    
    Args:
        fn: Path to the tar file
        split_ids: Dictionary mapping split names to lists of video IDs
        dirs: Directory structure for splits
        
    Returns:
        Dictionary mapping split names to lists of processed file IDs
    """
    # Initialize tracking for processed files
    processed_files = {split: [] for split in split_ids.keys()}
    
    # Get file path without extension
    file_path = os.path.splitext(fn)[0]
    
    # Check if it's a tar file that needs extraction
    if fn.endswith('.tar') and not os.path.isdir(file_path):
        if not untar_file(fn):
            return processed_files
    
    # Skip if directory doesn't exist
    if not os.path.isdir(file_path):
        print(f"Warning: Directory {file_path} does not exist or is not a directory")
        return processed_files
        
    # Get video IDs from the extracted directory
    try:
        video_ids = os.listdir(file_path)
    except Exception as e:
        print(f"Error listing directory {file_path}: {e}")
        return processed_files
    
    # Process each split
    for split, ids in split_ids.items():
        # Filter videos that belong to this split
        split_vids = [vid for vid in video_ids if vid in ids]
        
        # Move each video to its destination
        for vid in split_vids:
            source_path = os.path.join(file_path, vid)
            dest_path = os.path.join(dirs['src'][split], vid)
            
            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, dest_path)
                    processed_files[split].append(vid)
                except Exception as e:
                    print(f"Error moving {vid}: {e}")
    
    return processed_files


def create_validation_split(train_dir: str, val_dir: str) -> Tuple[List[str], List[str]]:
    """
    Create a validation set by moving files from train to validation directory.
    
    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory
        
    Returns:
        Tuple of (remaining_train_files, validation_files)
    """
    # Set seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Get and shuffle training files
    train_files = os.listdir(train_dir)
    random.shuffle(train_files)
    
    # Calculate the split index (90% train, 10% validation)
    split_idx = int(len(train_files) * (1 - VALIDATION_PERCENTAGE))
    
    # Select validation files
    val_files = train_files[split_idx:]
    remaining_train = train_files[:split_idx]
    
    # Move validation files to validation directory
    for file_name in val_files:
        source_path = os.path.join(train_dir, file_name)
        dest_path = os.path.join(val_dir, file_name)
        shutil.move(source_path, dest_path)
    
    print(f"Split complete: {len(remaining_train)} files in training, {len(val_files)} files in validation")
    return remaining_train, val_files


def create_split_csv(split_dir: str, original_csv: str, output_csv: str, columns: List[str]) -> None:
    """
    Create a new CSV file containing only the files present in the split directory.
    
    Args:
        split_dir: Directory containing the split files
        original_csv: Path to the original CSV file
        output_csv: Path to save the new CSV file
        columns: Column names for the CSV
    """
    # Get files in the split directory
    try:
        split_files = set(os.listdir(split_dir))
    except Exception as e:
        print(f"Error accessing directory {split_dir}: {e}")
        return
    
    # Read and filter the original CSV
    try:
        df = pd.read_csv(original_csv, names=columns)
        filtered_df = df[df['video_id'].isin(split_files)]
        
        # Save the filtered CSV
        filtered_df.to_csv(output_csv, index=False)
        print(f"Created {output_csv} with {len(filtered_df)} entries")
    except Exception as e:
        print(f"Error creating CSV {output_csv}: {e}")


if __name__ == "__main__":
    # Set up dataset paths
    dataset = 'avspeech'
    data_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)
    dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)
    
    # Create directories if needed
    attempt_makedirs(data_dir)

    # Get dataset configuration
    dataset_config = utils.DATASET_CONFIGS[dataset]
    splits = dataset_config['splits']
    
    # Add validation split if needed
    if 'val' not in splits:
        splits.append('val')

    # Prepare directory structure
    dirs, splits = utils.prepare_directory_structure(data_dir, splits, video=True)

    # Define CSV configuration
    split_fn = lambda x: os.path.join(dataset_dir, f'avspeech_{x}.csv')
    csv_splits = ['train', 'test']
    split_columns = ['video_id', 'start_segment', 'end_segment', 'x_coord', 'y_coord']

    # Load video IDs for each split
    split_ids = {}
    for split in csv_splits:
        try:
            df = pd.read_csv(split_fn(split), names=split_columns)
            split_ids[split] = df['video_id'].tolist()
        except Exception as e:
            print(f"Error loading CSV for {split}: {e}")
            split_ids[split] = []
    
    # Get all tar files
    tar_fns = sorted(glob.glob(os.path.join(dataset_dir, 'clips', '*')))
    
    if not tar_fns:
        print(f"Warning: No tar files found in {os.path.join(dataset_dir, 'clips')}")
    
    # Process tar files in parallel
    print("Starting parallel processing of tar files...")
    all_processed_files = {split: [] for split in split_ids.keys()}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tar files for processing
        future_to_tar = {
            executor.submit(process_tar_file, fn, split_ids, dirs): fn 
            for fn in tar_fns
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_tar):
            try:
                tar_fn = future_to_tar[future]
                results = future.result()
                
                # Track processed files
                for split, files in results.items():
                    all_processed_files[split].extend(files)
                    
                print(f"Processed {tar_fn}: {sum(len(files) for files in results.values())} files")
            except Exception as e:
                print(f"Error processing {future_to_tar[future]}: {e}")
    
    print("Parallel processing complete.")
    
    # Create validation split
    train_dir = dirs['src']['train']
    val_dir = dirs['src']['val']
    
    print("Creating validation split...")
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.listdir(train_dir):
        remaining_train, validation_files = create_validation_split(train_dir, val_dir)
    else:
        if not os.path.exists(train_dir):
            print(f"Training directory {train_dir} not found")
        elif not os.path.exists(val_dir):
            print(f"Validation directory {val_dir} not found")
        elif not os.listdir(train_dir):
            print(f"Training directory {train_dir} is empty")
        remaining_train, validation_files = [], []
    
    # Create new CSV files for each split
    print("Creating new CSV files for each split...")
    
    # Create train CSV
    create_split_csv(
        train_dir, 
        split_fn('train'), 
        os.path.join(dataset_dir, 'avspeech_train_new.csv'), 
        split_columns
    )
    
    # Create validation CSV
    if validation_files:
        try:
            train_df = pd.read_csv(split_fn('train'), names=split_columns)
            val_df = train_df[train_df['video_id'].isin(validation_files)]
            val_csv_path = os.path.join(dataset_dir, 'avspeech_val.csv')
            val_df.to_csv(val_csv_path, index=False)
            print(f"Created validation CSV at {val_csv_path} with {len(val_df)} entries")
        except Exception as e:
            print(f"Error creating validation CSV: {e}")
    
    # Create test CSV
    create_split_csv(
        dirs['src']['test'], 
        split_fn('test'), 
        os.path.join(dataset_dir, 'avspeech_test_new.csv'), 
        split_columns
    )
    
    print("Processing complete!")