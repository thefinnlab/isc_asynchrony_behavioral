import os, sys
import glob
import numpy as np
import pandas as pd
import subprocess
from praatio import textgrid as tgio
import argparse
import shutil 

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils
from preproc_utils import cut_stimulus_segments

def create_participant_df(df_preproc, df_segments):
	# these columns aren't needed here
	df_segments.drop(['checked', 'adjusted'], axis=1, inplace=True)

	# columns that we will will sample from the preprocessed per stimulus
	if 'entropy_group' in df_preproc.columns:
		sample_columns = ['Word_Written', 'Punctuation', 'Onset', 'Offset', 'Duration', 'entropy_group', 'accuracy_group']
	else:
		sample_columns = ['Word_Written', 'Punctuation', 'Onset', 'Offset', 'Duration']

	# for the practice trials we use the exception because we're hacky like that
	catch_columns = ['Word_Written', 'Punctuation', 'Onset', 'Offset', 'Duration']
	split_preprocessed = np.split(df_preproc, df_segments['word_index'])[:-1]

	for i, df in enumerate(split_preprocessed):
		df = df[sample_columns].reset_index(drop=True)

		# normalize to the start of the trial
		if i == 0:
			df[['Norm_Onset', 'Norm_Offset']] = df[['Onset', 'Offset']]
			df_segments.loc[i, ['entropy_group', 'accuracy_group']] = None
		else:
			df[['Norm_Onset', 'Norm_Offset']] = df[['Onset', 'Offset']] - df['Onset'].iloc[0]

			try:
				df_segments.loc[i-1, ['entropy_group', 'accuracy_group']] = df.loc[0, ['entropy_group', 'accuracy_group']]
			except:
				df_segments.loc[i-1, ['entropy_group', 'accuracy_group']] = None

		# turn to a json to for handling in javascript
		df_json = df.to_json(orient='records')
		df_segments.loc[i, 'word_info'] = df_json

	return df_segments

TASK_SUB_NUMS = {
	# number of subjects, number of orders
	'black': [200, 4],
	'wheretheressmoke': [150, 3],
	'howtodraw': [150, 3]
}

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_version', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-s', '--subject', type=str, default=None)
	parser.add_argument('-stim_type', '--stim_type', type=str, default='audio')
	p = parser.parse_args()
	
	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	presentation_orders_dir = os.path.join(stim_dir, 'presentation_orders', p.experiment_version, p.task)

	if p.stim_type == 'video':
		stim_fn = glob.glob(os.path.join(stim_dir, 'video', f'{p.task}*.mp4'))
		jspsych_out_dir = os.path.join(presentation_orders_dir, 'jspsych-video')
	else:
		stim_fn = glob.glob(os.path.join(stim_dir, 'audio', f'{p.task}*.wav'))
		jspsych_out_dir = os.path.join(presentation_orders_dir, 'jspsych')

	assert (len(stim_fn) == 1) 

	stim_fn = stim_fn[0]
	stim_out_dir = os.path.join(stim_dir, f'cut_{p.stim_type}', p.experiment_version, p.task, p.subject)

	utils.attempt_makedirs(jspsych_out_dir)
	utils.attempt_makedirs(stim_out_dir)

	# load preprocessed transcript and find indices that are to be predicted
	df_preproc_fn = os.path.join(presentation_orders_dir, 'preproc', f'{p.subject}_task-{p.task}.csv')
	df_preproc = pd.read_csv(df_preproc_fn)

	# Create sequential pairs
	candidate_idxs = np.where(df_preproc['NWP_Candidate'].to_numpy())[0] # First get indices
	candidate_idxs = np.concatenate([[0], candidate_idxs], axis=0) # Add the first item for the first cut
	segments = np.vstack((candidate_idxs[:-1], candidate_idxs[1:])).T # Stack and make pairs

	# Grab the last index and create a crop to the end of the data
	last_idxs = [candidate_idxs[-1], len(df_preproc)-1]
	segments = np.vstack((segments, last_idxs))

	print (segments)

	segment_indices = segments.tolist()
	
	# perform audio segmenting for the current subject
	out_fns, df_segments = cut_stimulus_segments(df_preproc, task=p.task, stim_fn=stim_fn, stim_out_dir=stim_out_dir, segment_indices=segment_indices, stim_type=p.stim_type)
	df_participant = create_participant_df(df_preproc, df_segments)

	# now write the file for jspsych to use
	jspsych_fn = os.path.join(jspsych_out_dir, f'{p.subject}_task-{p.task}')
	df_participant.to_csv(f'{jspsych_fn}.csv', index=False)
	df_participant.to_json(f'{jspsych_fn}.json', orient='records')

	# Hack to copy the video files
	if p.task != 'nwp_practice_trial' and p.stim_type == 'video':
		# Grab number of subjects / number of order for current task
		num_subs, num_orders = TASK_SUB_NUMS[p.task]
		sub_ids = np.arange(num_subs)

		# Get current subject number 
		current_sub = int(p.subject.split('-')[-1])
		
		# Grab matching subject orders --> increment by the current subject number
		sub_ids = sub_ids[::num_orders] + current_sub

		for sid in sub_ids:
			# Skip the current subject
			if sid == current_sub:
				continue

			# create the current subject name
			out_subject = f'sub-{str(sid).zfill(5)}'
			out_subject_dir = stim_out_dir.replace(p.subject, out_subject)

			# Write the participant file to the subject name
			out_fn = jspsych_fn.replace(p.subject, out_subject)
			df_participant.to_csv(f'{out_fn}.csv', index=False)
			df_participant.to_json(f'{out_fn}.json', orient='records')

			# Copy the cut files to the current output directory
			shutil.copytree(stim_out_dir, out_subject_dir, dirs_exist_ok=True)	