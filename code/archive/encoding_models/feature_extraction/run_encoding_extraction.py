import sys, os
import glob
import json
import argparse
import numpy as np
import pandas as pd 
import nibabel as nib
import librosa

sys.path.append('../../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import encoding, nlp

import torch

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# type of analysis we're running --> linked to the name of the regressors
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-window', '--window_size', type=int, default=25)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	print (f'Running feature extraction: task - {p.task} model - {p.model_name}', flush=True)

	output_dir = os.path.join(BASE_DIR, 'derivatives/regressors', p.dataset, p.task, p.model_name)

	if p.task in ['wheretheressmoke', 'howtodraw', 'odetostepfather']:
		gentle_dir = os.path.join(BASE_DIR, 'stimuli/gentle', p.task)
	else:
		gentle_dir = os.path.join(DATASETS_DIR, p.dataset, 'stimuli/gentle', p.task)

	utils.attempt_makedirs(output_dir)

	# get filenames
	audio_fn = os.path.join(gentle_dir, 'a.wav')
	transcript_fn = os.path.join(gentle_dir, 'align.json')

	# get the length of the stimulus spaced out by the frequency of the TRs
	task_idx = utils.DATASETS[p.dataset]['tasks'].index(p.task)
	n_trs = utils.DATASETS[p.dataset]['n_trs'][task_idx]
	tr_times = np.arange(0, n_trs*TR, TR)

	# load the gentle transcript
	if p.model_name in encoding.ENCODING_FEATURES['visual']:

		sys.exit(0)

		# video_clips = VideoClips([stim_fn], clip_length_in_frames=1, frames_between_clips=1)
		# n_frames = video_clips.num_clips()
		# video_fps = video_clips.video_fps[0]

	elif p.model_name in encoding.ENCODING_FEATURES['audio']:

		audio, sr = librosa.load(audio_fn, sr=None)
		audio = torch.tensor(audio)

		if audio.ndim == 1:
			audio = audio[np.newaxis]

	elif p.model_name in encoding.ENCODING_FEATURES['language']:

		if transcript_fn:
			df_transcript = encoding.load_gentle_transcript(
				transcript_fn=transcript_fn, 
				start_offset=None #stim_times[0] if stim_times else None
			)
		else:
			print (f'No transcript for {p.dataset} {p.task}')
			sys.exit(0)
	else:
		print (f'Error -- model not listed in tommy_utils.encoding')
		sys.exit(0)

	####################################
	######## FEATURE EXTRACTION ########
	####################################

	if p.model_name == 'phoneme':

		# create a phoneme feature space based on CMU phonemes
		times, features = encoding.create_phoneme_features(df_transcript)

	# word vector model extraction
	elif p.model_name in nlp.WORD_MODELS.keys():

		# load the word vector model
		word_model = nlp.load_word_model(model_name=p.model_name, cache_dir=CACHE_DIR)
		times, features = encoding.create_word_features(df_transcript, word_model=word_model)

	# masked language models extraction
	elif p.model_name in nlp.MLM_MODELS_DICT.keys():

		# load the masked language model
		mlm_tokenizer, mlm_model = nlp.load_mlm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
		times, features = encoding.create_transformer_features(df_transcript, mlm_tokenizer, mlm_model, window_size=p.window_size, add_punctuation=False)

	# causal language model extraction
	elif p.model_name in nlp.CLM_MODELS_DICT.keys():

		# load clm model
		clm_tokenizer, clm_model = nlp.load_clm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
		times, features = encoding.create_transformer_features(df_transcript, clm_tokenizer, clm_model, window_size=p.window_size, add_punctuation=False)
	
	# cnn features (potentially expand to trasformers)
	elif p.model_name in encoding.VISION_MODELS_DICT or p.model_name in nlp.MULTIMODAL_MODELS_DICT:

		# frames over fps gives us seconds, 1/hz is number of intervals
		times = np.arange(0, n_frames / video_fps, 1/video_fps)

		image_info = (times, video_clips)
		times, features = encoding.create_vision_features(image_info, model_name=p.model_name, batch_size=16)

	### start of audio features
	elif p.model_name == 'spectral':

		# create spectral feature space using melspectrogram
		times, features = encoding.create_spectral_features(audio, sr=sr)
	
	elif p.model_name in encoding.AUDIO_MODELS_DICT:

		# audio lengths over sampling rate gives us seconds, 1/hz is number of intervals
		times = np.arange(0, audio.shape[-1]/sr, 1/sr)
		sys.exit(0)
		pass
	else:
		sys.exit(0)
	
	# the list covers the case of vision features otherwise layers of language features
	if isinstance(features, list) or features.ndim > 2:
		for i, layer in enumerate(features):
			layer_number = f'layer-{str(i+1).zfill(3)}'
			out_fn = os.path.join(output_dir, f'task-{p.task}_model-{p.model_name}_{layer_number}.npy')

			downsampled_layer = encoding.lanczosinterp2D(data=layer, oldtime=times, newtime=tr_times)
			print (f'Features size {downsampled_layer.shape} after downsampling')
			
			np.save(out_fn, downsampled_layer)
	else:
		# downsample the feature set
		downsampled_features = encoding.lanczosinterp2D(data=features, oldtime=times, newtime=tr_times)
		
		# now save the features out
		out_fn = os.path.join(output_dir, f'task-{p.task}_features-{p.model_name}.npy')
		np.save(out_fn, downsampled_features)