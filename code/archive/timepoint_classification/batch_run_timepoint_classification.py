import sys, os
import glob
import argparse
import subprocess

from itertools import product

sys.path.append('../utils/')

import dataset_utils as utils
from config import *

PARTITION='preemptable'
TIME = '24:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '8G'
GPU_INFO = ''

NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'dbic'

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	if p.dataset == 'deniz-readinglistening':
		dataset_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc')
	else:
		dataset_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc-smooth')
	
	results_dir = os.path.join(BASE_DIR, 'derivatives/results', p.dataset)

	dsq_dir = os.path.join(SUBMIT_DIR, p.dataset, 'dsq')
	joblist_dir = os.path.join(SUBMIT_DIR, p.dataset, 'joblists')
	logs_dir = os.path.join(LOGS_DIR, p.dataset)

	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblist_dir)
	utils.attempt_makedirs(logs_dir)

	# get subjects excluding their html files
	sub_dirs = sorted(glob.glob(os.path.join(dataset_dir, 'sub*[!html]')))
	sub_list = [os.path.basename(d) for d in sub_dirs]

	# get task list
	task_list = utils.DATASETS[p.dataset]['tasks']
	task_list = set(task_list).difference(['wheretheressmoke']) # don't use wheretheressmoke

	####################################
	### PREPARE THE DATA FOR SCRIPT ####
	####################################

	sessions = ['reading', 'listening']

	# now prep for the script
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_timepoint_classification.py')
	job_string = f'{DSQ_MODULES} srun python -u {script_fn}'
	job_num = 0

	for train_session, test_session in product(sessions, repeat=2):
		# pair each subject and task
		for sub, task in product(sub_list, task_list):

			print (sub, task)

			cmd = ''.join([
				f'{job_string} -d {p.dataset}  -s {sub} -t {task} -train_session {train_session} -test_session {test_session} -n_proc {CPUS_PER_TASK}']
			)

			all_cmds.append(cmd)
			job_num += 1

	if not all_cmds:
		print (f'No files needing timepoint classification', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_timepoint_classification_joblist.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_timepoint_classification'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} {GPU_INFO} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --account={ACCOUNT} {NODE_LIST} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)