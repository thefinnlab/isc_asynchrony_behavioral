import sys, os
import glob
import argparse
import subprocess
import argparse

sys.path.append('../../utils/')

import dataset_utils as utils
from config import *

PARTITION='preemptable'
TIME = '5:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '16G'
GPU_INFO = ''

# PARTITION = 'v100_12'
# GPU_INFO = '--gres=gpu:1'

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

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

	print (sub_list)

	####################################
	### PREPARE THE DATA FOR SCRIPT ####
	####################################

	MODEL_NAMES = ['spectral', 'phoneme', 'word2vec', 'gpt2-xl'] #'gpt2-xl']

	test_tasks = ['wheretheressmoke']
	test_tasks = ' '.join(test_tasks)

	# now prep for the script
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_permutation_tests.py')
	job_string = f'{DSQ_MODULES} srun python -u {script_fn}'
	job_num = 0

	iteration = 1
	main_model = MODEL_NAMES[-1]

	for sub in sub_list:

		features = ' '.join(MODEL_NAMES)
		files_exist = any(glob.glob(os.path.join(results_dir, sub, main_model, '*permutations*')))

		# only run dsq for regressions that haven't been run yet
		if p.overwrite or not files_exist:

			cmd = ''.join([
				f'{job_string} -d {p.dataset} -test {test_tasks} -s {sub} -m {main_model} -i {str(iteration).zfill(5)} -f {features} -o {p.overwrite}']
			)

			all_cmds.append(cmd)
			job_num += 1
	
	if not all_cmds:
		print (f'No files needing encoding models - overwrite if you want to redo', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_permutations_joblist.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_permutation_tests'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} {GPU_INFO} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
