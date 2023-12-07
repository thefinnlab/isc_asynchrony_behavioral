import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
from itertools import product
import subprocess

sys.path.append('../utils/')

from config import *
from nlp_utils import CLM_MODELS_DICT

PARTITION = 'preemptable'
TIME = '12:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'

if __name__ == '__main__':

	OVERWRITE = True

	# grab the tasks
	task_dirs = sorted(glob.glob(os.path.join(STIM_DIR, 'preprocessed', '*')))
	task_names = [os.path.basename(d) for d in task_dirs if os.path.isdir(d)]

	# remove practice trial and example trial from the list of tasks
	task_names = [task for task in task_names if task not in ['nwp_practice_trial', 'example_trial']] 
	
	model_names = sorted(CLM_MODELS_DICT.keys())
	window_sizes = [25, 50, 100]
	top_ns = [1, 5, 10]

	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_get_clm_predictions.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0
	
	for i, (task, model, window) in enumerate(product(task_names, model_names, window_sizes)):

		if i not in rerun_jobs:
			continue

		out_dir = os.path.join(DERIVATIVES_DIR, 'model-predictions', task, model, f'window-size-{window}')
		out_fn = os.path.join(out_dir, f'task-{task}_model-{model}_window-size-{window}_top-{top_ns[0]}.csv')
		file_exists = os.path.exists(out_fn)

		if OVERWRITE or not file_exists:
			cmd = ''.join([
				f'{job_string} -t {task} -m {model} -w {window} -n {" ".join(str(v) for v in top_ns)}']
			)

			all_cmds.append(cmd)
			job_num += 1

	if not all_cmds:
		print (f'No files needing predictions - overwrite if you want to redo predictions', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(JOBLIST_DIR, f'run_get_clm_predictions.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")

	dsq_base_string = f'dsq_run_get_clm_predictions'
	dsq_batch_fn = os.path.join(DSQ_DIR, dsq_base_string)
	dsq_out_dir = os.path.join(LOGS_DIR, dsq_base_string)
	array_fmt_width = len(str(job_num))
	
	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	# subprocess.run('module load dSQ', shell=True)
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.out "
		f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)