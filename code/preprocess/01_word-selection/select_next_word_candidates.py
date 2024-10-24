import os, sys, glob
import json
from operator import itemgetter
import re
import numpy as np
import pandas as pd
import shutil
from praatio import textgrid as tgio

sys.path.append('../../utils/')

from config import *
from preproc_utils import create_word_prediction_df, clean_hyphenated_words, clean_named_entities, dataframe_to_textgrid, get_word_frequency

if __name__ == '__main__':

	task = sys.argv[1] #'black' # replace this string with those above
	overwrite = False

	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	gentle_dir = os.path.join(stim_dir, 'gentle')
	preproc_dir = os.path.join(stim_dir,'preprocessed')
	task_out_dir = os.path.join(preproc_dir, task)
	backup_dir = os.path.join(task_out_dir, 'src')

	audio_fn = glob.glob(os.path.join(stim_dir, 'audio', f'*{task}*.wav'))[0]

	if not os.path.exists(task_out_dir):
		os.makedirs(task_out_dir)
	
	if not os.path.exists(backup_dir):
		os.makedirs(backup_dir)

	# loads the alignment file and parses variables of interest into dataframe
	df_task_raw_fn = os.path.join(task_out_dir, f'{task}_transcript-raw.csv')
	df_preproc_fn = os.path.join(task_out_dir, f'{task}_transcript-preprocessed')

	# if the file doesn't exists or we want to overwrite the file
	if not os.path.exists(f'{df_preproc_fn}.csv') or overwrite:
		df_task_raw = create_word_prediction_df(os.path.join(gentle_dir, task, 'align.json'), fill_missing_times=True)
		df_task_raw.to_csv(df_task_raw_fn, index=False)
	else:
		print (f'File exists - exiting')
		df_preproc = pd.read_csv(f'{df_preproc_fn}.csv')
		df_preproc.to_json(f'{df_preproc_fn}.json', orient='records')

		shutil.copyfile(f'{df_preproc_fn}.csv', os.path.join(backup_dir, f'{os.path.basename(df_preproc_fn)}.csv'))
		shutil.copyfile(f'{df_preproc_fn}.json', os.path.join(backup_dir, f'{os.path.basename(df_preproc_fn)}.json'))
		sys.exit(0)

	# now we clean hyphenated words --> either collapse them or separate the hyphen better
	df_preproc = clean_hyphenated_words(df_task_raw)
	
	# clean named entities --> remove any pronouns/names/cities etc from prediction candidates
	df_preproc = clean_named_entities(df_preproc)

	# now we select the candidates:
	# - for practice/example trial, we have set candidates
	# - otherwise candidates are those that aren't named entities or stop-words
	if task == 'nwp_practice_trial':
		practice_indices = df_preproc['Word_Written'].isin(['practice', 'recording', 'word'])
		df_preproc['NWP_Candidate'] = practice_indices
	elif task == 'example_trial':
		example_indices = df_preproc['Word_Written'].isin(['fox', 'lazy'])
		df_preproc['NWP_Candidate'] = example_indices
	elif task == 'myfirstdaywiththeyankees':
		exclude_indices = df_preproc['Word_Written'].isin(['know'])
		df_preproc.loc[exclude_indices, 'NWP_Candidate'] = False
	else:
		df_preproc['NWP_Candidate'] = pd.Series(df_preproc['Named_Entity'] == False) & \
			pd.Series(df_preproc['Stop_Word'] == False) & \
			pd.Series(df_preproc['Digit'] == False) & \
			pd.Series(df_preproc['Case'] == 'success') 

	### Add in word frequency information
	df_preproc = get_word_frequency(df_preproc)

	#save to file
	df_preproc.to_csv(f'{df_preproc_fn}.csv', index=False)
	df_preproc.to_json(f'{df_preproc_fn}.json', orient='records') #, lines=True) #, index=False)

	praat_fn = os.path.join(preproc_dir, task, f'{task}_transcript-praat.TextGrid')

	if not os.path.exists(praat_fn):
		tg = dataframe_to_textgrid(df_preproc, audio_fn)
		tg.save(praat_fn, 'long_textgrid', True)

	shutil.copyfile(f'{df_preproc_fn}.csv', os.path.join(backup_dir, f'{os.path.basename(df_preproc_fn)}.csv'))
	shutil.copyfile(f'{df_preproc_fn}.json', os.path.join(backup_dir, f'{os.path.basename(df_preproc_fn)}.json'))
	shutil.copyfile(praat_fn, os.path.join(backup_dir, f'{os.path.basename(praat_fn)}'))
