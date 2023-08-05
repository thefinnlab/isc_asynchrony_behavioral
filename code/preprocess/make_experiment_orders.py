import sys, os
import numpy as np
import pandas as pd
import json

CLOSING_URL = "https://app.prolific.co/submissions/complete?cc=C1LRSG9N"

EXPERIMENT_VERSION = 'pilot-version-04'
EXPERIMENT_NAME = "next-word-prediction"

# set directories
base_dir = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavioral/'
experiment_dir = '/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/utils/'

orders_dir = os.path.join(base_dir, 'stimuli', 'presentation_orders', EXPERIMENT_VERSION)

global_parameters = {
	'output_path': "/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/data/",
	'closing_url': CLOSING_URL,
	'num_experiments': 1,
	"current_experiment": 0
}

## SETUP EXPERIMENT PARAMETERS

modality_list = ['audio', 'text', 'audio-text']

url_base = "https://rcweb.dartmouth.edu/~f003rjw/jspsych_experiments/experiments/isc_asynchrony_behavior/"

experiment_parameters = {
	"experiment_name": EXPERIMENT_NAME,
	"experiment_version": EXPERIMENT_VERSION,
	"experiment_url": os.path.join(url_base, "experiments/next-word-prediction/next-word-prediction.html"),
	"practice_transcript": os.path.join(url_base, "stimuli/preprocessed/nwp_practice_trial/nwp_practice_trial_transcript_preprocessed.json"),
	"practice_audio": os.path.join(url_base, "stimuli/audio/nwp_practice_trial_audio.wav"), 
}

# get list of directories
tasks = [ item for item in os.listdir(orders_dir) if os.path.isdir(os.path.join(orders_dir, item)) ]

fns = []

sub_counter = 1
	
for task in tasks:
	task_dir = os.path.join(orders_dir, task)
	task_parameter_files = sorted(os.listdir(os.path.join(orders_dir, task)))
	
	for fn in task_parameter_files:

		for modality in modality_list:
			experiment_info_dir = os.path.join(base_dir, 'experiments', EXPERIMENT_NAME, 'experiment_orders', EXPERIMENT_VERSION, modality)

			if not os.path.exists(experiment_info_dir):
				os.makedirs(experiment_info_dir)

			sub_parameters = experiment_parameters.copy()
			
			if 'json' not in fn:
				continue
			
			sub_parameters.update({
				"stimulus_transcript": os.path.join(url_base, 'stimuli/presentation_orders/', EXPERIMENT_VERSION, task, fn),
				"stimulus_audio": os.path.join(url_base, 'stimuli/audio/', f'{task}_audio.wav'),
				"stimulus_modality": modality
			})
			
			out_json = [global_parameters, sub_parameters]
			out_fn = os.path.join(experiment_info_dir, f'sub-{str(sub_counter).zfill(5)}_experiment-information.json')
			
			with open(out_fn, 'w') as f:
				json.dump(out_json, f)

			fns.append(out_fn.replace(base_dir, url_base))

		sub_counter += 1

df = pd.DataFrame(fns, columns=['subject_fns'])
df['used'] = pd.Series(dtype='str')

df.to_csv(os.path.join(experiment_dir, f'{EXPERIMENT_NAME}_{EXPERIMENT_VERSION}.csv'), index=False)
