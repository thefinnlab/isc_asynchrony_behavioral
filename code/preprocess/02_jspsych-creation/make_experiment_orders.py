import sys, os
import numpy as np
import pandas as pd
import glob
import json

CLOSING_URL = "https://app.prolific.co/submissions/complete?cc=C1LRSG9N"
EXPERIMENT_NAME = "next-word-prediction"

URL_BASE = "https://rcweb.dartmouth.edu/~f003rjw/jspsych_experiments/experiments/isc_asynchrony_behavior/"

## SETUP EXPERIMENT PARAMETERS
MODALITY_LIST = [
	'audio',
	'text',
	# 'audio-text'
]

if __name__ == '__main__':

	EXPERIMENT_VERSION = sys.argv[1] 
	task = sys.argv[2] #'test'

	# set directories
	base_dir = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/'
	experiment_dir = '/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/utils/experiment_meta/'
	orders_dir = os.path.join(base_dir, 'stimuli', 'presentation_orders', EXPERIMENT_VERSION)

	global_parameters = {
		'output_path': "/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/data/",
		'closing_url': CLOSING_URL,
		'num_experiments': 1,
		"current_experiment": 0
	}

	experiment_parameters = {
		"experiment_name": EXPERIMENT_NAME,
		"experiment_version": EXPERIMENT_VERSION,
		"experiment_url": os.path.join(URL_BASE, "experiments/next-word-prediction/next-word-prediction.html")
	}
	
	# get list of directories
	# tasks = [ item for item in os.listdir(orders_dir) if os.path.isdir(os.path.join(orders_dir, item)) ]

	fns = []

	sub_counter = 1
		
	# for task in tasks:
	task_parameter_files = sorted(glob.glob(os.path.join(orders_dir, task, 'jspsych', '*.json')))
	# task_parameter_files = sorted(os.listdir(os.path.join(orders_dir, task)))
	
	for fn in task_parameter_files:

		fn = os.path.basename(fn)

		for modality in MODALITY_LIST:
			experiment_info_dir = os.path.join(base_dir, 'experiments', EXPERIMENT_NAME, 'experiment_orders', EXPERIMENT_VERSION, modality)

			if not os.path.exists(experiment_info_dir):
				os.makedirs(experiment_info_dir)

			sub_parameters = experiment_parameters.copy()
			
			if 'json' not in fn:
				continue
			
			sub_parameters.update({
				"stimulus_info": os.path.join(URL_BASE, 'stimuli/presentation_orders/', EXPERIMENT_VERSION, task, 'jspsych', fn),
				"stimulus_modality": modality,
				"practice_info": os.path.join(URL_BASE, 'stimuli/presentation_orders/', EXPERIMENT_VERSION, "nwp_practice_trial/jspsych/practice_task-nwp_practice_trial.json"),
			})
			
			out_json = [global_parameters, sub_parameters]
			out_fn = os.path.join(experiment_info_dir, f'sub-{str(sub_counter).zfill(5)}_experiment-information.json')
			
			with open(out_fn, 'w') as f:
				json.dump(out_json, f)

			fns.append(out_fn.replace(base_dir, URL_BASE))

		sub_counter += 1

	df = pd.DataFrame(fns, columns=['subject_fns'])
	df['used'] = pd.Series(dtype='str')

	df.to_csv(os.path.join(experiment_dir, EXPERIMENT_NAME, f'{EXPERIMENT_VERSION}.csv'), index=False)
