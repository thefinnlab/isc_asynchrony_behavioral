from pydub import AudioSegment
from pydub.playback import play
import os, sys
import numpy as np
import pandas as pd
import subprocess
from praatio import textgrid as tgio

sys.path.append('../utils/')

from config import *
from preproc_utils import update_dataframe_from_praat, dataframe_to_textgrid

def get_cut_times(df, start_idx, end_idx):
	
	onset = df.iloc[start_idx]['Onset']
	offset = df.iloc[end_idx]['Onset']
	
	duration = offset - onset
	
	return onset, offset, duration

if __name__ == '__main__':

	task = 'black'
	
	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	preproc_dir = os.path.join(stim_dir, 'preprocessed')

	audio_fn = os.path.join(stim_dir, 'audio', f'{task}_audio.wav')
	audio_out_dir = os.path.join(stim_dir, 'cut_audio', task)

	if not os.path.exists(audio_out_dir):
		os.makedirs(audio_out_dir)

	# load the stimulus and fine the length in time
	stim = AudioSegment.from_file(os.path.join(stim_dir, 'audio', f'{task}_audio.wav'))
	stim_length = stim.duration_seconds

	# load preprocessed transcript and find indices that are to be predicted
	df_preproc_fn = os.path.join(preproc_dir, task, f'{task}_transcript-preprocessed.csv')
	df_preproc = pd.read_csv(df_preproc_fn)
	prediction_idxs = np.where(df_preproc['NWP_Candidate'])[0]

	## Segments are defined as follows
	##  - Start = where a previous segment left of --> will contain the prior segment's predicted word
	##  - Stop = ending right before a word prediction
	## Therefore adjusting the end of one will cause a shift in the subsequent segment time

	# create dataframe that accompanies written audio segments
	# get segment file if it exists
	df_segments_fn = os.path.join(preproc_dir, task, f'{task}_transcript-segments.csv')
	praat_fn = os.path.join(preproc_dir, task, f'{task}_transcript-praat.TextGrid')

	if os.path.exists(df_segments_fn):
		df_segments = pd.read_csv(df_segments_fn)
	else:
		df_segments = pd.DataFrame(columns=['filename', 'word_index', 'critical_word', 'checked', 'adjusted'])

	# if a textgrid file exists, we open it and use it in to adjust the times
	if os.path.exists(praat_fn):
		tg = tgio.openTextgrid(praat_fn, False)
		df_preproc = update_dataframe_from_praat(df_preproc, tg)
	else:
		tg = dataframe_to_textgrid(df_preproc, audio_fn)
		tg.save(praat_fn, 'long_textgrid', True)

	for i, curr_idx in enumerate(prediction_idxs):
		# if we're on the first index we use the start of the file

		if i == 0:
			_, offset, _ = get_cut_times(df_preproc, 0, curr_idx)
			onset = 0
			duration = offset
		elif i == len(prediction_idxs):
			onset, _, _ = get_cut_times(df_preproc, curr_idx, curr_idx)
			duration = stim_length - onset
		else:
			prev_idx = prediction_idxs[i-1]
			onset, _, duration = get_cut_times(df_preproc, prev_idx, curr_idx)
		
		out_fn = os.path.join(audio_out_dir, f'{task}_segment-{str(i+1).zfill(5)}.wav')
		cmd = f'ffmpeg -y -ss {onset} -t {duration} -i {audio_fn} {out_fn}'
		subprocess.run(cmd, shell=True)

		# if the segments file does not exist
		if not os.path.exists(df_segments_fn):
			df_segments.loc[len(df_segments)] = {
				'filename': out_fn,
				'word_index': curr_idx,
				'critical_word': df_preproc.loc[curr_idx]['Word_Written'],
				'checked': 0,
				'adjusted': 0
			}
	
	# find if there were any adjusted indices
	adjusted_idxs = np.where(df_segments['adjusted'])[0]

	# if there are indices, we need to recheck both the adjusted segment and the subsequent segment 
	if len(adjusted_idxs):
		adjusted_idxs = set(adjusted_idxs).union(adjusted_idxs + 1).difference(len(df_segments))
		df_segments.at[adjusted_idxs, ['checked', 'adjusted']] = 0

	df_segments.to_csv(df_segments_fn, index=False)