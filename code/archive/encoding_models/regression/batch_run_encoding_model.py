import sys, os
import glob
import argparse
import subprocess
import argparse

sys.path.append('../../utils/')

import dataset_utils as utils
from config import *

PARTITION='preemptable'
TIME = '24:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '1-00:00:00'
CPUS_PER_TASK = 4
MEM_PER_CPU = '16G'
PARTITION = 'a100'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'test_a100'

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

	print (sub_list)

	####################################
	### PREPARE THE DATA FOR SCRIPT ####
	####################################

	BEHAVIORAL_TASKS = ['wheretheressmoke', 'howtodraw', 'odetostepfather']
	behavior_additions = sorted(set(BEHAVIORAL_TASKS).difference(['wheretheressmoke']))
	# all our models will have this
	
	if p.dataset == 'deniz-readinglistening':

		MODEL_NAMES = ['phoneme', 'word2vec', 'gpt2-xl']

		# remove wheretheressmoke validation task from train set
		train_tasks = utils.DATASETS[p.dataset]['tasks']
		train_tasks = sorted(set(train_tasks).difference(BEHAVIORAL_TASKS))

		# 2 runs of where theres smoke for deniz dataset
		test_tasks = [f"wheretheressmoke_run-{i}" for i in range(1,3)]
		test_tasks.extend(behavior_additions)
		
		# then separate into reading and listening sessions
		train_test_splits = []
		sessions = ['reading', 'listening']

		for ses in sessions:
			# add in motion features for the reading data
			if ses == 'reading':
				ses_features = ['motion'] + MODEL_NAMES
				ses_info = (ses, train_tasks, test_tasks, ses_features)
			elif ses == 'listening':
				ses_featuers = ['spectral'] + MODEL_NAMES
				ses_info = (ses, train_tasks, test_tasks, MODEL_NAMES)

			train_test_splits.append(ses_info)
	else:
		# huth moth tasks
		MODEL_NAMES = ['spectral', 'phoneme', 'word2vec', 'gpt2-xl']

		# remove wheretheressmoke validation task from train set
		train_tasks = utils.DATASETS[p.dataset]['tasks']
		train_tasks = sorted(set(train_tasks).difference(BEHAVIORAL_TASKS))

		# make test set each individual run of where theres smoke
		test_tasks = [f"wheretheressmoke_run-{i}" for i in range(1,6)]
		test_tasks.extend(behavior_additions)

		# put together
		train_test_splits = [(train_tasks, test_tasks, MODEL_NAMES)]

	# now prep for the script
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_encoding_model.py')
	job_string = f'{DSQ_MODULES} srun python -u {script_fn}'
	job_num = 0

	iteration = 1
	main_model = MODEL_NAMES[-1]

	for split_info in train_test_splits:
		# add in session if necessary
		if len(split_info) == 4:
			ses, train_tasks, test_tasks, features = split_info
			ses = f'-ses {ses}'
			features = ' '.join(features)
		else:
			train_tasks, test_tasks, features = split_info
			ses = f''
			features = ' '.join(features)

		print (f'Train tasks {train_tasks}')
		print (f'Test tasks {test_tasks}')

		train_tasks = ' '.join(train_tasks)
		test_tasks = ' '.join(test_tasks)

		for sub in sub_list:
			files_exist = any(glob.glob(os.path.join(results_dir, sub, main_model, '*')))

			# only run dsq for regressions that haven't been run yet
			if p.overwrite or not files_exist:

				cmd = ''.join([
					f'{job_string} -d {p.dataset}  -s {sub} -train {train_tasks} -test {test_tasks} {ses} -m {main_model} -i {str(iteration).zfill(5)} -f {features} -o {p.overwrite}']
				)

				all_cmds.append(cmd)
				job_num += 1

	if not all_cmds:
		print (f'No files needing encoding models - overwrite if you want to redo', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_encoding_model_joblist.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_encoding_model'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} {GPU_INFO} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --account={ACCOUNT} {NODE_LIST} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)