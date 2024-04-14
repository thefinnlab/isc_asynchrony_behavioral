import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
import torch
from torch.nn import functional as F

sys.path.append('../utils/')

from config import *
import nlp_utils as nlp

def preproc_to_input(df_preproc, idxs):
	'''
	Given the preprocessed dataframe, extract the transcript text
	over a set of indices to submit to a model
	'''
	
	# get the GT upcoming word to predict
	ground_truth_word = df_preproc.loc[idxs[-1] + 1, 'Word_Written']
	
	# get the segment to submit to the model
	df_segment = df_preproc.iloc[idxs]
	
	all_items = []
	
	for i, row in df_segment.iterrows():
		# sometimes there is punctuation, other times there is whitespace
		# we add in the punctuation as it helps the model but remove trailing
		# whitespaces
		item = ''.join(row[['Word_Written', 'Punctuation']])
		all_items.append(item.strip())
		
	all_items = ' '.join(all_items)
	
	return all_items, ground_truth_word

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-w', '--window_size', type=int, default=25)
	parser.add_argument('-n', '--top_n', type=int, nargs='+', default=5)
	p = parser.parse_args()

	print (f'Task: {p.task}')
	print (f'Window Size: {p.window_size}')

	out_dir = os.path.join(DERIVATIVES_DIR, 'model-predictions', p.task, p.model_name, f'window-size-{p.window_size}')
	logits_dir = os.path.join(out_dir, 'logits')

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if not os.path.exists(logits_dir):
		os.makedirs(logits_dir)

	# load the preprocessed file --> this has next-word-candidates selected
	stim_preprocessed_fn = os.path.join(STIM_DIR, 'preprocessed', p.task, f'{p.task}_transcript-preprocessed.csv')
	df_preproc = pd.read_csv(stim_preprocessed_fn)

	# # load a word-level model --> we use glove here
	# # first function downloads the model if needed, second function loads it as gensim format
	word_models = {model_name: nlp.load_word_model(model_name=model_name, cache_dir=CACHE_DIR) for model_name in nlp.WORD_MODELS.keys()}

	# load the causal language model
	print (f'Loading {p.model_name}', flush=True)
	tokenizer, model =  nlp.load_clm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
	print (f'Model loaded', flush=True)
	
	# add the first word to the dataframe --> we don't run NWP on this as there is no context
	# to condition, nor do we have humans do it
	df = nlp.create_results_dataframe()
	first_word = df_preproc.iloc[0]['Word_Written'].lower()
	df.loc[len(df)] = {'ground_truth_word': first_word}

	# set up variables to be used in the loop
	df_stack = {str(n): [df] for n in p.top_n}
	prev_probs = None

	# create a list of indices that we will iterate through to sample the transcript
	segments = nlp.get_segment_indices(n_words=len(df_preproc), window_size=p.window_size)

	for i, segment in enumerate(segments):
		# also keep track of the current ground truth word
		inputs, ground_truth_word = preproc_to_input(df_preproc, segment)
		
		# run the inputs through the model and get the predictions
		# this returns the predictive distribution and saves out the logits

		# if p.model_name == 'gpt2-xl' and p.window_size == 100:
		# 	logits_fn = os.path.join(logits_dir, f'{p.task}_window-size-{p.window_size}_logits-{str(i).zfill(5)}.pt')
		# else:
		# 	logits_fn = None

		probs = nlp.get_clm_predictions([inputs], model, tokenizer) #, out_fn=logits_fn)
		
		# now given the outputs of the model, run our stats of interest
		for n in p.top_n:
			segment_stats = nlp.get_model_statistics(ground_truth_word, probs, tokenizer, prev_probs=prev_probs, word_models=word_models, top_n=n)
			df_stack[str(n)].append(segment_stats)

		# now that we've run our stats, set the previous distribution to the one we just ran
		prev_probs = probs
		
		print (f'Processed segment {i+1}/{len(segments)}', flush=True)

	for n in p.top_n:
		df_results = pd.concat(df_stack[str(n)])
		df_results.to_csv(os.path.join(out_dir, f'task-{p.task}_model-{p.model_name}_window-size-{p.window_size}_top-{n}.csv'), index=False)
