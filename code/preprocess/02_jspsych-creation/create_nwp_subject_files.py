import sys, os
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import subprocess

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils
from preproc_utils import (
	get_quadrant_distributions, 
	get_consecutive_list_idxs, 
	sort_consecutive_constraint, 
	check_consecutive_spacing,
	random_chunks
	)

TIME = 240
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '4G'
PARTITION = 'standard'

# CREATING PARTICIPANT ORDERS FOR PRESENTATION
N_ORDERS = 3
N_PARTICIPANTS_PER_ITEM = 50
CONSECUTIVE_SPACING = 10
PASS_THRESHOLD = 50 # number of loops for sorting before restarting

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_name', type=str)
	parser.add_argument('-t', '--task', type=str)
	p = parser.parse_args()

	# set directories
	preproc_dir = os.path.join(BASE_DIR, 'stimuli', 'preprocessed')
	task_out_dir = os.path.join(BASE_DIR, 'stimuli', 'presentation_orders', p.experiment_name, p.task, 'preproc')
	submit_dir = os.path.join(SUBMIT_DIR, 'behavioral')

	utils.attempt_makedirs(task_out_dir)
	utils.attempt_makedirs(submit_dir)

	# IF WE'RE ON A PRACTICE TRIAL
	if p.task == 'nwp_practice_trial':
		print ("MAKING PRACTICE TRIAL")

		subject = 'practice'

		# load preprocessed transcript
		df_task_preproc_fn = os.path.join(preproc_dir, p.task, f'{p.task}_transcript-preprocessed')
		df_preproc = pd.read_csv(f'{df_task_preproc_fn}.csv')
		
		# write files for the current subject
		sub_fn = os.path.join(task_out_dir, f'{subject}_task-{p.task}')
		
		# df_preproc[['entropy_group', 'accuracy_group']] = None
		df_preproc.to_csv(f'{sub_fn}.csv', index=False)
		df_preproc.to_json(f'{sub_fn}.json', orient='records')

		subprocess.run(f'python cut_participant_audio.py -n {p.experiment_name} -t {p.task} -s {subject}', shell=True)

		sys.exit(0)
	else:
		# load selected word transcript
		df_task_selected_fn = os.path.join(preproc_dir, p.task, f'{p.task}_transcript-selected')
		df_selected = pd.read_csv(f'{df_task_selected_fn}.csv')

		df_divide_fn = os.path.join(preproc_dir, p.task, f'{p.task}_transcript-model-divided.csv')
		df_divide = pd.read_csv(df_divide_fn)

	# get the natural distribution of the words --> deviation threshold is the amount of error
	# we tolerate from the distribution in each order
	quadrant_distribution = get_quadrant_distributions(df_divide, df_divide.index).to_numpy()
	deviation_threshold = 0.04
	order_distributions = np.zeros((N_ORDERS, 4))

	# get baseline distribution of quadrants
	nwp_indices = sorted(np.where(df_selected['NWP_Candidate'])[0])

	# Find lists with consecutive items violating our constraint
	while not (np.allclose(quadrant_distribution, order_distributions, atol=deviation_threshold)):

		# randomly chunk all indices into N_ORDERS
		subject_experiment_orders = random_chunks(nwp_indices, len(nwp_indices)//N_ORDERS, shuffle=True)

		# sort the consecutive constraint
		pass_threshold, subject_experiment_orders = sort_consecutive_constraint(subject_experiment_orders, consecutive_spacing=CONSECUTIVE_SPACING, pass_threshold=PASS_THRESHOLD)

		if not pass_threshold:
			continue

		# now find distribution of each order --> make sure it's equal to natural distrubtion
		order_distributions = [get_quadrant_distributions(df_selected, order).to_numpy() for order in subject_experiment_orders]

		# sometimes the randomized order makes a quadrant be dropped --> reset and try again
		if not all([order.shape[-1] == 4 for order in order_distributions]):
			order_distributions = np.zeros((N_ORDERS, 4))

	print (f'Natural distribution')
	print (quadrant_distribution)

	print (f'Order distributions')
	print (order_distributions)
	
	# ensure that we didn't violate the consecutive constraint
	violating_lists = get_consecutive_list_idxs(subject_experiment_orders, consecutive_spacing=CONSECUTIVE_SPACING)
	assert (not any(violating_lists))

	# now get a list for each subject (tiling them so evenly distributed)
	subject_experiment_orders = subject_experiment_orders * N_PARTICIPANTS_PER_ITEM

	assert (len(subject_experiment_orders) == (N_PARTICIPANTS_PER_ITEM * N_ORDERS))

	# now we make the orders per subject and cut the audio
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'cut_participant_audio.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'

	for i, order in enumerate(subject_experiment_orders):

		# set the subject name
		subject = f'sub-{str(i+1).zfill(5)}'

		# find indices not selected for the current subject and set to false
		df_subject = df_selected.copy()
		unselected = np.setdiff1d(nwp_indices, order)
		df_subject['NWP_Candidate'].loc[unselected] = False

		# write files for the current subject
		sub_fn = os.path.join(task_out_dir, f'{subject}_task-{p.task}')
		df_subject.to_csv(f'{sub_fn}.csv', index=False)
		df_subject.to_json(f'{sub_fn}.json', orient='records')

		cmd = ''.join([f'{job_string} -n {p.experiment_name} -t {p.task} -s {subject}'])
		all_cmds.append(cmd)

	### make dsq dirs
	logs_dir = os.path.join(LOGS_DIR, 'behavioral')
	dsq_dir = os.path.join(submit_dir, 'dsq')
	jobslist_dir = os.path.join(submit_dir, 'joblists')

	utils.attempt_makedirs(logs_dir)
	utils.attempt_makedirs(jobslist_dir)
	utils.attempt_makedirs(dsq_dir)

	joblist_fn = os.path.join(jobslist_dir, f'{p.experiment_name}_task-{p.task}_cut_participant_audio.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_run_cut_participant_audio'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(i))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	# subprocess.run('module load dSQ', shell=True)
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.out "
		f"--time={TIME} --nodes={N_NODES} --partition={PARTITION} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)