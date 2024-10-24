import os, sys
import glob
import numpy as np
import pandas as pd
import subprocess
from praatio import textgrid as tgio

sys.path.append('../../utils/')

from config import *
from preproc_utils import update_dataframe_from_praat, dataframe_to_textgrid, get_cut_times, cut_audio_segments

if __name__ == '__main__':
	
	# task = 'black'
	task = sys.argv[1]
	
	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	preproc_dir = os.path.join(stim_dir, 'preprocessed')

	audio_fn = glob.glob(os.path.join(stim_dir, 'audio', f'{task}*.wav'))
	assert (len(audio_fn) == 1)

	audio_fn = audio_fn[0]

	audio_out_dir = os.path.join(stim_dir, 'cut_audio', 'src', task)

	if not os.path.exists(audio_out_dir):
		os.makedirs(audio_out_dir)

	# load preprocessed transcript and find indices that are to be predicted
	df_preproc_fn = os.path.join(preproc_dir, task, f'{task}_transcript-preprocessed')
	df_preproc = pd.read_csv(f'{df_preproc_fn}.csv')

	## Segments are defined as follows
	##  - Start = where a previous segment left of --> will contain the prior segment's predicted word
	##  - Stop = ending right before a word prediction
	## Therefore adjusting the end of one will cause a shift in the subsequent segment time

	# create dataframe that accompanies written audio segments
	# get segment file if it exists
	df_segments_fn = os.path.join(preproc_dir, task, f'{task}_transcript-segments.csv')
	praat_fn = os.path.join(preproc_dir, task, f'{task}_transcript-praat.TextGrid')

	# if a textgrid file exists, we open it and use it in to adjust the times
	if os.path.exists(praat_fn):
		tg = tgio.openTextgrid(praat_fn, False)
		df_preproc = update_dataframe_from_praat(df_preproc, tg)
		print ('Updating from existing textgrid', flush=True)

		df_preproc.to_csv(f'{df_preproc_fn}.csv')
		df_preproc.to_json(f'{df_preproc_fn}.json', orient='records')
	else:
		tg = dataframe_to_textgrid(df_preproc, audio_fn)
		tg.save(praat_fn, 'long_textgrid', True)

	# we load the previous segments file and update it after cutting audio, otherwise create the segments file
	# if os.path.exists(df_segments_fn):
	# 	df_segments = pd.read_csv(df_segments_fn)
	# 	_ = cut_audio_segments(df_preproc, task=task, audio_fn=audio_fn, audio_out_dir=audio_out_dir)
	# else:
	_, df_segments = cut_audio_segments(df_preproc, task=task, audio_fn=audio_fn, audio_out_dir=audio_out_dir)
	
	# # find if there were any adjusted indices
	# adjusted_idxs = np.where(df_segments['adjusted'])[0]

	# # if there are indices, we need to recheck both the adjusted segment and the subsequent segment 
	# if len(adjusted_idxs):
	# 	adjusted_idxs = set(adjusted_idxs).union(adjusted_idxs + 1).difference(len(df_segments))
	# 	df_segments.at[adjusted_idxs, ['checked', 'adjusted']] = 0

	df_segments.to_csv(df_segments_fn, index=False)