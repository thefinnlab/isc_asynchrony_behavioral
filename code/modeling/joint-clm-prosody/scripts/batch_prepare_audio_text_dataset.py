import sys, os
import subprocess
from itertools import product
import argparse

from config import *
import utils

DATASET_INFO = {
	'gigaspeech': ['train', 'validation', 'test'],
	'libritts-r': ['train-clean-360', 'dev-clean', 'test-clean'],
	'tedlium': ['train'], #, 'validation', 'test'],
	'peoples-speech': ['train'], #, 'validation', 'test'],
	'pfka-moth-stories': ['black', 'wheretheressmoke', 'howtodraw']
}

PARTITION='preemptable'
TIME = '2-23:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '1-00:00:00'
CPUS_PER_TASK = 8
MEM_PER_CPU = '16G'
PARTITION = 'v100_preemptable'
GPU_INFO = '--gres=gpu:1'
ACCOUNT = 'dbic' #'test_a100'

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-s', '--split', type=str, default=None)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	# DATASETS = ['libritts-r']

	# make directories
	dsq_dir = os.path.join(SUBMIT_DIR, 'dsq')
	joblist_dir = os.path.join(SUBMIT_DIR, 'joblists')
	logs_dir = os.path.join(LOGS_DIR)

	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblist_dir)
	utils.attempt_makedirs(logs_dir)

	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'prepare_audio_text_dataset.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0

	if p.split:
		splits = [p.split]
	else:
		splits = DATASET_INFO[p.dataset]

	print (splits)
		
	for split in splits:
		cmd = f"{job_string} -d {p.dataset} -s {split} -o {p.overwrite}"
		all_cmds.append(cmd)
		job_num += 1

	if not all_cmds:
		print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_prepare_audio_text_dataset.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_{p.dataset}_prepare_audio_text_dataset'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --account={ACCOUNT} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
