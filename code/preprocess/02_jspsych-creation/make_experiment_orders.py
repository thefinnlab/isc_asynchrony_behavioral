import sys, os
import numpy as np
import pandas as pd
import glob
import json
import argparse

CLOSING_URL = "https://app.prolific.co/submissions/complete?cc=C1LRSG9N"
EXPERIMENT_NAME = "next-word-prediction"

URL_BASE = "https://rcweb.dartmouth.edu/~f003rjw/jspsych_experiments/experiments/isc_asynchrony_behavior/"

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_version', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-modality_list', '--modality_list', type=str, nargs='+', default=['video', 'audio', 'text'])
	p = parser.parse_args()

	# set directories
	base_dir = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/'
	experiment_dir = '/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/utils/experiment_meta/'
	orders_dir = os.path.join(base_dir, 'stimuli', 'presentation_orders', p.experiment_version)

	global_parameters = {
		'output_path': "/dartfs/rc/lab/F/FinnLab/tommy/jspsych_experiments/data/",
		'closing_url': CLOSING_URL,
		'num_experiments': 1,
		"current_experiment": 0
	}

	experiment_parameters = {
		"experiment_name": EXPERIMENT_NAME,
		"experiment_version": p.experiment_version,
		"task_name": p.task,
		"experiment_url": os.path.join(URL_BASE, "experiments/next-word-prediction/next-word-prediction.html")
	}
	
	# get list of directories
	# tasks = [ item for item in os.listdir(orders_dir) if os.path.isdir(os.path.join(orders_dir, item)) 
	if p.modality_list[0] == 'video':
		json_file_dir = 'jspsych-video'
	else:
		json_file_dir = 'jspsych'
	
	fns = []
	sub_counter = 1

	task_parameter_files = sorted(glob.glob(os.path.join(orders_dir, p.task, json_file_dir, '*.json')))

	for fn in task_parameter_files:

		fn = os.path.basename(fn)

		for modality in p.modality_list:

			experiment_info_dir = os.path.join(base_dir, 'experiments', EXPERIMENT_NAME, 'experiment_orders', p.experiment_version, p.task, modality)

			if not os.path.exists(experiment_info_dir):
				os.makedirs(experiment_info_dir)

			sub_parameters = experiment_parameters.copy()
			
			if 'json' not in fn:
				continue

			sub_parameters.update({
				"stimulus_info": os.path.join(URL_BASE, 'stimuli/presentation_orders/', p.experiment_version, p.task, json_file_dir, fn),
				"stimulus_modality": modality,
				"practice_info": os.path.join(URL_BASE, 'stimuli/presentation_orders/', p.experiment_version, "nwp_practice_trial/jspsych/practice_task-nwp_practice_trial.json"),
			})
			
			out_json = [global_parameters, sub_parameters]
			out_fn = os.path.join(experiment_info_dir, f'sub-{str(sub_counter).zfill(5)}_experiment-information.json')
			
			with open(out_fn, 'w') as f:
				json.dump(out_json, f)

			fns.append(out_fn.replace(base_dir, URL_BASE))

		sub_counter += 1

	df = pd.DataFrame(fns, columns=['subject_fns'])
	df['used'] = pd.Series(dtype='str')

	modalities = '-'.join(p.modality_list)
	df.to_csv(os.path.join(experiment_dir, EXPERIMENT_NAME, f'{p.experiment_version}-{p.task}_{modalities}.csv'), index=False)
