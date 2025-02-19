import sys, os
import glob
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import subprocess

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils

TIME = 240
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '4G'
PARTITION = 'standard'

NUM_ORDERS = {
	'black': 4,
	'wheretheressmoke': 3,
	'howtodraw': 3,
}

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--experiment_name', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-stim_type', '--stim_type', type=str, default='video')
	p = parser.parse_args()

	# set directories
	preproc_dir = os.path.join(BASE_DIR, 'stimuli', 'preprocessed')
	task_files_dir = os.path.join(BASE_DIR, 'stimuli', 'presentation_orders', p.experiment_name, p.task, 'preproc')
	submit_dir = os.path.join(SUBMIT_DIR, 'behavioral')

	utils.attempt_makedirs(task_files_dir)
	utils.attempt_makedirs(submit_dir)

	### make dsq dirs
	logs_dir = os.path.join(LOGS_DIR, 'behavioral')
	dsq_dir = os.path.join(submit_dir, 'dsq')
	jobslist_dir = os.path.join(submit_dir, 'joblists')

	utils.attempt_makedirs(logs_dir)
	utils.attempt_makedirs(jobslist_dir)
	utils.attempt_makedirs(dsq_dir)

	# now we make the orders per subject and cut the audio
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'cut_participant_stimuli.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	num_jobs = 0

	subject_fns = sorted(glob.glob(os.path.join(task_files_dir, '*.csv')))

	# Copy same files across subjects instead of producing separately for each subject
	if p.stim_type == 'video':
		num_orders = NUM_ORDERS[p.task]

	for fn in subject_fns:
		subject = os.path.basename(fn).split('_')[0]

		cmd = ''.join([f'{job_string} -n {p.experiment_name} -t {p.task} -s {subject} -stim_type {p.stim_type}'])
		all_cmds.append(cmd)
		num_jobs += 1 

	joblist_fn = os.path.join(jobslist_dir, f'{p.experiment_name}_task-{p.task}_cut_participant_stimuli.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_run_cut_participant_stimuli'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(num_jobs))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	# subprocess.run('module load dSQ', shell=True)
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --nodes={N_NODES} --partition={PARTITION} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)