import sys, os
import glob
import json
import argparse
import numpy as np
import pandas as pd 
import nibabel as nib
import librosa
from itertools import combinations

sys.path.append('../../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import nlp, encoding

import torch

TASK_INFO = {
	'tasks': [
		'alternateithicatom', 'avatar', 'legacy', 'odetostepfather', 'souls',
		'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'undertheinfluence', #'life',
		'exorcism', 'fromboyhoodtofatherhood', 'sloth', 'stagefright', 'tildeath',
		'adollshouse', 'adventuresinsayingyes', 'buck', 'haveyoumethimyet', 'inamoment', 'theclosetthatateeverything',
		'eyespy', 'hangtime', 'itsabox', 'swimmingwithastronauts', 'thatthingonmyarm', 'wheretheressmoke'
	],
	'n_trs': [
		354, 378, 410, 414, 360, 
		365, 368, 433, 314, #440,
		478, 357, 448, 304, 334,
		252, 402, 343, 507, 215, 325,
		389, 334, 365, 395, 444, 300
	]
}

def get_substitute_word_vectors(df_transcript, df_substitute, features, tokenizer, model, window_size=25, bidirectional=False):
	
	segments = nlp.get_segment_indices(n_words=len(df_transcript), window_size=window_size, bidirectional=bidirectional)
	
	all_idx_embeddings = []
	
	# we process row by row as to not disrupt the transcript itself
	for i, row in df_substitute.iterrows():
		
		print (f'Processing substitution: {i+1}/{len(df_substitute)}')
		
		# get the subsitute word for the current row
		word_idx = row['word_index']
		segment = segments[word_idx]
		df_seg = df_transcript.loc[segment].copy()
		
		# now sub in the word at the current index
		df_seg.loc[word_idx, 'word'] = row['top_pred']
		
		# get the prepared input for the word embedding extraction
		inputs = nlp.transcript_to_input(df_seg, segment, add_punctuation=False)
		
		model_embeddings = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=-1).squeeze()
		all_idx_embeddings.append((word_idx, model_embeddings))
	
	idxs, embeddings = zip(*all_idx_embeddings)
	embeddings = torch.stack(embeddings)
	embeddings = torch.moveaxis(embeddings, 1, 0)
	
	### TLB NOTE THIS CAN CAUSE INEQUIVALENT NUMBER OF CHANGES ACROSS MODELS/HUMANS
	substitute_features = features.copy()
	substitute_features[:, idxs, :] = embeddings
	
	print (f'Altered {len(idxs)} total words')
	
	return idxs, substitute_features

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

	# we output to the behavioral directory to avoid interference
	output_dir = os.path.join(BASE_DIR, 'derivatives', 'regressors', p.dataset, p.task, 'behavioral', p.model_name)
	gentle_dir = os.path.join(DATASETS_DIR, 'huth-moth', 'stimuli', 'gentle', p.task)

	utils.attempt_makedirs(output_dir)

	# get filenames
	audio_fn = os.path.join(gentle_dir, 'a.wav')
	transcript_fn = os.path.join(gentle_dir, 'align.json')

	# get the length of the stimulus spaced out by the frequency of the TRs
	task_idx = TASK_INFO['tasks'].index(p.task)
	n_trs = TASK_INFO['n_trs'][task_idx]
	tr_times = np.arange(0, n_trs*TR, TR)

	# get model features
	if p.model_name in encoding.ENCODING_FEATURES['language']:

		if transcript_fn:
			df_transcript = encoding.load_gentle_transcript(
				transcript_fn=transcript_fn, 
				start_offset=None #stim_times[0] if stim_times else None
			)
		else:
			print (f'No transcript for {p.dataset} {p.task}')
			sys.exit(0)
	else:
		print (f'Error')
		sys.exit(0)

	####################################
	######## FEATURE EXTRACTION ########
	####################################

	if p.model_name in nlp.MLM_MODELS_DICT.keys():

		# load the masked language model
		tokenizer, model = nlp.load_mlm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
		# times, features = encoding.create_transformer_features(df_transcript, tokenizer, model, add_punctuation=False)

	# causal language model extraction
	elif p.model_name in nlp.CLM_MODELS_DICT.keys():

		# load clm model
		tokenizer, model = nlp.load_clm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
		# times, features = encoding.create_transformer_features(df_transcript, tokenizer, model, add_punctuation=False)

	## now read the behavior results
	## MAKE ADJUSTMENTS TO MODEL FEATURES FROM BEHAVIOR ########
	behavior_results_fn = os.path.join(BASE_DIR, f'derivatives/results/behavioral/task-{p.task}_group-analyzed-behavior_human-model-lemmatized.csv')
	behavior_results = pd.read_csv(behavior_results_fn)

	# compare the model within human/audio
	conditions = [p.model_name, 'audio', 'text']

	all_conditions_info = []

	for cond in conditions:
		df_condition = df_transcript.copy()

		# get words for current condition and reset indices
		df_condition_words = behavior_results[behavior_results['modality'] == cond].reset_index(drop=True)

		df_condition.loc[df_condition_words['word_index'], 'word'] = df_condition_words['top_pred'].tolist()

		times, features = encoding.create_transformer_features(df_condition, tokenizer, model, window_size=p.window_size, add_punctuation=False)

		if cond != p.model_name:
			cond = f'human-{cond}'
		else:
			cond = f'model'

		for i, layer in enumerate(features):
			layer_number = f'layer-{str(i+1).zfill(3)}'
			out_fn = os.path.join(output_dir, f'task-{p.task}_model-{p.model_name}_{layer_number}_{cond}-predicted.npy')

			downsampled_layer = encoding.lanczosinterp2D(data=layer, oldtime=times, newtime=tr_times)
			print (f'Features size {downsampled_layer.shape} after downsampling')
			
			np.save(out_fn, downsampled_layer)

	# 	# now subsitute words and get their embeddings
	# 	idxs, condition_word_embeddings = get_substitute_word_vectors(df_transcript, df_condition_words, features, tokenizer, model)
	# 	all_conditions_info.append((cond, idxs, condition_word_embeddings))

	# # unpack and make sure all conditions have the same indices as substituted
	# conditions, cond_idxs, condition_word_embeddings = zip(*all_conditions_info)
	# all_idxs_equal = all([np.array_equal(*comb) for comb in combinations(cond_idxs, r=2)])

	# if not all_idxs_equal:
	# 	print (f'Problem with array indices {p.model}')
	# 	sys.exit(0)

	# # go through each condition and save out
	# for cond, _, embeddings in all_conditions_info:

	# 	if cond != p.model_name:
	# 		cond = f'human-{cond}'
	# 	else:
	# 		cond = f'model'

		# for i, layer in enumerate(embeddings):
		# 	layer_number = f'layer-{str(i+1).zfill(3)}'
		# 	out_fn = os.path.join(output_dir, f'task-{p.task}_model-{p.model_name}_{layer_number}_{cond}-predicted.npy')

		# 	downsampled_layer = encoding.lanczosinterp2D(data=layer, oldtime=times, newtime=tr_times)
		# 	print (f'Features size {downsampled_layer.shape} after downsampling')
			
		# 	np.save(out_fn, downsampled_layer)