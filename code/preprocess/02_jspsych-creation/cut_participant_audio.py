from pydub import AudioSegment
from pydub.playback import play
import os, sys
import numpy as np
import pandas as pd
import subprocess
from praatio import textgrid as tgio
import argparse

sys.path.append('../utils/')

from config import *
from preproc_utils import cut_audio_segments

def create_participant_df(df_preproc, df_segments):
	# these columns aren't needed here
	df_segments.drop(['checked', 'adjusted'], axis=1, inplace=True)

	# columns that we will will sample from the preprocessed per stimulus
	sample_columns = ['Word_Written', 'Punctuation', 'Onset', 'Offset', 'Duration']
	split_preprocessed = np.split(df_preproc, df_segments['word_index'])[:-1]

	for i, df in enumerate(split_preprocessed):
		df = df[sample_columns].reset_index(drop=True)

		# normalize to the start of the trial
		if i == 0:
			df[['Norm_Onset', 'Norm_Offset']] = df[['Onset', 'Offset']]
		else:
			df[['Norm_Onset', 'Norm_Offset']] = df[['Onset', 'Offset']] - df['Onset'].iloc[0]

		# turn to a json to for handling in javascript
		df_json = df.to_json(orient='records')
		df_segments.loc[i, 'word_info'] = df_json

	return df_segments

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_name', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-s', '--subject', type=str, default=None)
	p = parser.parse_args()
	
	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	presentation_orders_dir = os.path.join(stim_dir, 'presentation_orders', p.experiment_name, p.task)
	audio_fn = os.path.join(stim_dir, 'audio', f'{p.task}_audio.wav')
	audio_out_dir = os.path.join(stim_dir, 'cut_audio', p.experiment_name, p.task, p.subject)

	if not os.path.exists(os.path.join(presentation_orders_dir, 'jspsych')):
		try:
			os.makedirs(os.path.join(presentation_orders_dir, 'jspsych'))
		except:
			pass

	if not os.path.exists(audio_out_dir):
		try:
			os.makedirs(audio_out_dir)
		except:
			pass

	# load preprocessed transcript and find indices that are to be predicted
	df_preproc_fn = os.path.join(presentation_orders_dir, 'preproc', f'{p.subject}_task-{p.task}.csv')
	df_preproc = pd.read_csv(df_preproc_fn)

	# perform audio segmenting for the current subject
	out_fns, df_segments = cut_audio_segments(df_preproc, task=p.task, audio_fn=audio_fn, audio_out_dir=audio_out_dir)
	df_participant = create_participant_df(df_preproc, df_segments)

	# now write the file for jspsych to use
	jspsych_fn = os.path.join(presentation_orders_dir, 'jspsych', f'{p.subject}_task-{p.task}')
	df_participant.to_csv(f'{jspsych_fn}.csv', index=False)
	df_participant.to_json(f'{jspsych_fn}.json', orient='records')