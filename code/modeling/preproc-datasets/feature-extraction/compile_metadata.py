import os
import glob
import argparse
import shutil
from tqdm import tqdm

sys.path.append('../')

import utils

def compile_features(path):
    """Load data from temp JSON and replace file paths with actual data."""
    data = utils.load_json(path)
    
    # Map of paths to load and their destination keys
    mapping = {
        'text_tokens_path': 'text_tokens',
        'attention_mask_path': 'attention_mask',
        'prominence_path': 'prominence',
        'boundary_path': 'boundary'
    }

    if 'audio_features_path' in data:
        mapping['audio_features_path'] = 'audio_features'
    
    if 'video_features_path' in data:
        mapping['video_features_path'] = 'video_features'

    # Load data from paths and replace paths with data
    for k, v in mapping.items():
        if k in data:
            data[v] = list(torch.load(data[k]))
            del data[k]

    return data

def compile_metadata(args, metadata, temp_dir):
	'''
	Path to the metadata directory
	'''

	# Find all files that have been preprocessed
	all_fns = sorted(os.listdir(temp_dir))
	to_process_files = [fn.replace('_processed.json', '') for fn in all_fns]

	# If we don't want to overwrite and metadata exists
	if not args.overwrite and metadata:

		# Search for existing files
	    existing_files = [item['base_name'] for item in metadata]

	    # Find set difference between the basenames --> only files that need to be added
	    to_process_files = set(to_process_files).difference(existing_files)
	    to_process_files = sorted(to_process_files)

	    print(f"Found {len(existing_files)} existing transcripts", flush=True)

	print(f"Processing {len(to_process_files)} files into metadata", flush=True)

    # Go through each file to process
	for fn in tqdm(to_process_files):
		temp_json_path = utils.get_temp_json_path(temp_dir, fn)
		data = compile_features(temp_json_path)
        metadata.append(data)

    return metadata


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# parser.add_argument('--base_dir')
	parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--split', type=str, default=None,
	                  help='Which split to process')

    ### Model names
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=list(AUDIO_MODELS.keys()) + [None], 
                        help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default=None, choices=list(VIDEO_MODELS.keys()) + [None], 
                        help='Video model to use, or None to skip video processing')

    ### Video
    parser.add_argument('--video', type=int, default=0, help="Whether this is an AV dataset")
	parser.add_argument('-o', '--overwrite', type=int, default=0)

    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val', 'test']
    
    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    for split in splits:
        # Setup cache directories
        model_combo = f"{args.text_model}"

        if args.audio_model:
            model_combo += f"-{args.audio_model}"

        if args.video_model:
            model_combo += f"-{args.video_model}"

        # Create cache for our features and a temp directory for writing progress
        cache_dir = os.path.join(args.output_dir, split, 'features', model_combo)
        temp_dir = os.path.join(cache_dir, 'temp')
        errors_dir = os.path.join(cache_dir, 'errors')

        # Metadata paths
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        error_metadata_path = os.path.join(cache_dir, 'error_metadata.json')

        # Load or create metadata --> if doesn't exist, will return an empty list
        metadata = utils.load_json(metadata_path)
        metadata = compile_metadata(args, metadata, temp_dir)
        utils.save_json(metadata_path, metadata)

        # Repeat same process for errors information
        error_metadata = utils.load_json(error_metadata_path)
        error_metadata = compile_metadata(args, error_metadata, errors_dir)
        utils.save_json(error_metadata_path, error_metadata)