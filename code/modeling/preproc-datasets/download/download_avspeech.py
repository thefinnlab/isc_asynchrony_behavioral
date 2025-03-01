# First run the following command
'''
aria2c --seed-time=0 --max-overall-download-limit=10M 
	--file-allocation=none --dir=./avspeech/ 
	https://academictorrents.com/download/b078815ca447a3e4d17e8a2a34f13183ec5dec41.torrent
'''
import os
import csv
import shutil
import tarfile
import random
import string
import glob

def extract_all_tars(extract_dir):
    """Extract all tar files from xaa.tar to xpj.tar"""
    # Generate all possible file names from xaa.tar to xpj.tar
    # This covers a-z for the first position and a-j for the second position
    first_chars = string.ascii_lowercase[:24]  # a to x
    second_chars = string.ascii_lowercase[:10]  # a to j
    
    tar_files = []
    for first in first_chars:
        for second in second_chars:
            tar_name = f"x{first}{second}.tar"
            if os.path.exists(tar_name):
                tar_files.append(tar_name)
    
    # Alternative approach using glob
    if not tar_files:
        tar_files = glob.glob("x??.tar")
    
    for tar_file in tar_files:
        print(f"Extracting {tar_file}...")
        try:
            with tarfile.open(tar_file, 'r') as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"Error extracting {tar_file}: {e}")

def organize_files(extract_dir, train_dir, test_dir):
    """Organize files into train and test directories based on CSV files"""
    # Read train.csv
    train_files = read_csv_files('train.csv')
    
    # Read test.csv
    test_files = read_csv_files('test.csv')
    
    # Move files to appropriate directories
    for file_name in train_files:
        src_path = os.path.join(extract_dir, file_name)
        dst_path = os.path.join(train_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    for file_name in test_files:
        src_path = os.path.join(extract_dir, file_name)
        dst_path = os.path.join(test_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

def read_csv_files(csv_file):
    """Read file names from a CSV file"""
    files = []
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found!")
        return files
    
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row:  # Check if row is not empty
                files.append(row[0])  # Assume first column contains file names
    
    return files

def create_validation_split(train_dir, val_dir):
    """Create a validation set by moving 10% of training data"""
    # Set a consistent seed for reproducibility
    random.seed(42)
    
    # Get all files in train directory
    train_files = os.listdir(train_dir)
    
    # Shuffle the files
    random.shuffle(train_files)
    
    # Calculate split index (90/10 split)
    split_idx = int(len(train_files) * 0.9)
    
    # Select validation files (10% of training data)
    val_files = train_files[split_idx:]
    
    # Move validation files to validation directory
    for file_name in val_files:
        src_path = os.path.join(train_dir, file_name)
        dst_path = os.path.join(val_dir, file_name)
        shutil.move(src_path, dst_path)
    
    print(f"Split complete: {len(train_files) - len(val_files)} files in training, {len(val_files)} files in validation")

def main():
    # Create necessary directories if they don't exist
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, 'src')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    extract_dir = os.path.join(base_dir, 'extracted_files')
    
    for directory in [output_dir, train_dir, val_dir, test_dir, extract_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Extract all tar files
    print("Extracting tar files...")
    extract_all_tars(extract_dir)
    
    # Step 2: Organize files into train and test based on CSV files
    print("Organizing files based on CSV files...")
    organize_files(extract_dir, train_dir, test_dir)
    
    # Step 3: Create validation set from train data
    print("Creating validation set...")
    create_validation_split(train_dir, val_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()