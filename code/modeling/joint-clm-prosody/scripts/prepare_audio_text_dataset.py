import os, sys
import glob
import argparse

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/utils/')
sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/')

from config import *
from src.data.components.audio_text_dataset import AudioTextDataset

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-s', '--split', type=str)
	parser.add_argument('-audio_model_name', '--audio_model_name', type=str, default='wav2vec2')
	parser.add_argument('-text_model_name', '--text_model_name', type=str, default='gpt2')
	p = parser.parse_args()

	if p.dataset == 'gigaspeech':
		dataset_dir = os.path.join(DATASETS_DIR, p.dataset, 'm')
		cache_dir = os.path.join(SCRATCH_DIR, p.dataset, 'm')
	else:
		sys.exit(0)

	# create datasets
	dataset = AudioTextDataset(
		dataset_dir=dataset_dir,
		cache_dir=cache_dir,
		audio_model_name=p.audio_model_name, 
		text_model_name=p.text_model_name, 
		split=p.split,
	)

	dataset.preprocess_data()