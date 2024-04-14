import sys, os
import glob
import argparse
import subprocess

sys.path.append('../../utils/')

import dataset_utils as utils
import encoding_utils as encoding
from config import *

PARTITION = 'standard'
TIME = 240
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '32G'

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	MODEL_NAMES = ['gpt2', 'gpt2-xl']

	# get tasks from gentle dir
	behavior_fns = sorted(glob.glob(os.path.join(BASE_DIR, 'derivatives/results/behavioral/*.csv')))
	TASKS = [os.path.basename(fn).split('_')[0].split('-')[-1] for fn in behavior_fns]
	print (f'Preparing the following tasks: {TASKS}')

	regressors_dir = os.path.join(BASE_DIR, 'derivatives', 'regressors', p.dataset)
	dsq_dir = os.path.join(SUBMIT_DIR, p.dataset, 'dsq')
	joblist_dir = os.path.join(SUBMIT_DIR, p.dataset, 'joblists')
	logs_dir = os.path.join(LOGS_DIR, p.dataset)

	utils.attempt_makedirs(regressors_dir)
	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblist_dir)
	utils.attempt_makedirs(logs_dir)

	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_word_substitution.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0
	
	for task in TASKS:
		for model_name in MODEL_NAMES:

			model_exists = any(glob.glob(os.path.join(regressors_dir, task, 'behavioral', f'{model_name}', '*')))
			
			# only run dsq for features that haven't been extracted yet
			# only case where not entering loop is when file exists and overwrite is false
			if p.overwrite or not model_exists:
				cmd = ''.join([
					f'{job_string} -d {p.dataset} -t {task} -m {model_name} -o {p.overwrite}']
				)
				
				all_cmds.append(cmd)
				job_num += 1

	if not all_cmds:
		print (f'No model needing substitution - overwrite if you want to redo extraction', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_word_substitution_joblist.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_word_substitution'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
