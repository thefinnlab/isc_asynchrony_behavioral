import sys, os
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import subprocess

sys.path.append('../utils/')

from config import *
from preproc_utils import create_balanced_orders, get_consecutive_list_idxs, sort_consecutive_constraint, check_consecutive_spacing

TIME = 240
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '4G'

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_name', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-p', '--percent_sampled', type=float, default=0.25) # percentage of items to sample for each subject
	parser.add_argument('-c', '--n_participants_per_item', type=int, default=25) # number of times items are seen across subjects
	parser.add_argument('-i', '--consecutive_spacing', type=int, default=2) # number of times items are seen across subjects
	p = parser.parse_args()

	# set directories
	preproc_dir = os.path.join(BASE_DIR, 'stimuli', 'preprocessed')
	task_out_dir = os.path.join(BASE_DIR, 'stimuli', 'presentation_orders', p.experiment_name, p.task, 'preproc')

	if not os.path.exists(task_out_dir):
		os.makedirs(task_out_dir)

	# load preprocessed transcript
	df_task_preproc_fn = os.path.join(preproc_dir, p.task, f'{p.task}_transcript-preprocessed')
	df_preproc = pd.read_csv(f'{df_task_preproc_fn}.csv')

	# IF WE'RE ON A PRACTICE TRIAL
	if p.task == 'nwp_practice_trial':
		print ("MAKING PRACTICE TRIAL")
		subject = 'practice'
		# write files for the current subject
		sub_fn = os.path.join(task_out_dir, f'{subject}_task-{p.task}')
		df_preproc.to_csv(f'{sub_fn}.csv', index=False)
		df_preproc.to_json(f'{sub_fn}.json', orient='records')

		subprocess.run(f'python cut_participant_audio.py -n {p.experiment_name} -t {p.task} -s {subject}', shell=True)

		sys.exit(0)

	# find indices for presentation and set number of items each subject sees
	nwp_indices = np.where(df_preproc['NWP_Candidate'])[0]
	n_items_per_subject = round(len(nwp_indices) * p.percent_sampled)

	# create experiment structure for subjects --> sort the indices
	subject_experiment_orders = create_balanced_orders(items=nwp_indices, n_elements_per_subject=n_items_per_subject, use_each_times=p.n_participants_per_item)
	subject_experiment_orders = list(map(sorted, subject_experiment_orders))

	# Find lists with consecutive items violating our constraint
	idxs = get_consecutive_list_idxs(subject_experiment_orders, consecutive_spacing=p.consecutive_spacing)
	subject_experiment_orders = sort_consecutive_constraint(subject_experiment_orders, consecutive_spacing=p.consecutive_spacing)

	# Test again once we have completed resorting
	idxs = get_consecutive_list_idxs(subject_experiment_orders, consecutive_spacing=p.consecutive_spacing)
	print (f'Lists violating consecutive index constraint: {100*(len(idxs))/len(subject_experiment_orders)}%')

	uniq, counts = np.unique(subject_experiment_orders, return_counts=True)
	print (f'All counts per word: {np.sum(counts >= p.n_participants_per_item) / len(counts)*100}%')

	counts = Counter(tuple(o) for o in subject_experiment_orders)
	unique_orders = np.sum([v for k, v in counts.items()]) / len(counts)

	print (f'Unique orders: {unique_orders*100}%')

	orders_meeting_consecutive = np.sum([check_consecutive_spacing(order, consecutive_spacing=p.consecutive_spacing) for order in subject_experiment_orders]) / len(subject_experiment_orders)
	print (f'Consecutive constraint: {orders_meeting_consecutive*100}%')

	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'cut_participant_audio.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'

	for i, order in enumerate(subject_experiment_orders):

		# set the subject name
		subject = f'sub-{str(i+1).zfill(5)}'

		# find indices not selected for the current subject and set to false
		df_subject = df_preproc.copy()
		unselected = np.setdiff1d(nwp_indices, order)
		df_subject['NWP_Candidate'].loc[unselected] = False

		# write files for the current subject
		sub_fn = os.path.join(task_out_dir, f'{subject}_task-{p.task}')
		df_subject.to_csv(f'{sub_fn}.csv', index=False)
		df_subject.to_json(f'{sub_fn}.json', orient='records')

		cmd = ''.join([f'{job_string} -n {p.experiment_name} -t {p.task} -s {subject}'])
		all_cmds.append(cmd)

	joblist_fn = os.path.join(JOBLIST_DIR, f'{p.experiment_name}_task-{p.task}_cut_participant_audio.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_run_cut_participant_audio'
	dsq_batch_fn = os.path.join(DSQ_DIR, dsq_base_string)
	dsq_out_dir = os.path.join(LOGS_DIR, dsq_base_string)
	array_fmt_width = len(str(i))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	# subprocess.run('module load dSQ', shell=True)
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.out "
		f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
