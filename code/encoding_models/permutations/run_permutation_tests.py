import sys, os
import glob
import argparse
import numpy as np

sys.path.append('../../utils/') 

import dataset_utils as utils
from tommy_utils import statistics
from config import *

from himalaya.scoring import correlation_score

N_PROC = 16
N_PERMS = 1000

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Dataset to use (and task name within datasets
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-s', '--sub', type=str) 

	parser.add_argument('-test', '--test_tasks', type=str, nargs='+')
	parser.add_argument('-ses', '--session', type=str, default=None) # session is mostly used to stratify tasks

	# type of analysis we're running --> linked to the name of the regressors
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-i', '--iteration', type=str)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	print (f'Running permutations on {p.dataset}, {p.sub}, {p.session}', flush=True)
	print (f'Test tasks: {p.test_tasks}', flush=True)

	if p.dataset == 'deniz-readinglistening':
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc/', p.sub)
	else:
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc-smooth/', p.sub)

	results_dir = os.path.join(BASE_DIR, 'derivatives', 'results', p.dataset, p.sub, p.model_name)

	#############################################################
	#### Load predictions & actual data, create distribution ####
	#############################################################

	for task in p.test_tasks:

		if p.dataset == 'deniz-readinglistening':
			if p.session:
				sub_data_fn = sorted(glob.glob(os.path.join(data_dir, f'*{p.session}*{task}*hyperaligned.npy')))
				sub_results_fn = sorted(glob.glob(os.path.join(results_dir, f'*{p.session}*{task}*prediction*ground-truth*.npy')))
			else:
				sub_data_fn = sorted(glob.glob(os.path.join(data_dir, f'*{task}*hyperaligned.npy')))
				sub_results_fn = sorted(glob.glob(os.path.join(results_dir, f'*{task}prediction*ground-truth*.npy')))
		else:
			sub_data_fn = sorted(glob.glob(os.path.join(data_dir, f'*{task}*hyperaligned.npy')))
			sub_results_fn = sorted(glob.glob(os.path.join(results_dir, f'*{task}*prediction*ground-truth*.npy')))

		## Load the data
		if 'wheretheressmoke' in task:
			assert (len(sub_data_fn) == 5 or len(sub_data_fn) == 2)
			Y_true = np.mean([np.load(fn) for fn in sub_data_fn], axis=0)
			Y_pred = np.mean([np.load(fn) for fn in sub_results_fn], axis=0)
		else:
			assert (len(sub_data_fn) == 1)
			Y_true = np.load(sub_data_fn[0])
			Y_pred = np.load(sub_results_fn[0])

		print (f'Starting permutations for {task}', flush=True)

		print (f'Y_true shape: {Y_true.shape}')
		print (f'Y_pred shape: {Y_pred.shape}')
		
		# run permutation test
		permutations = statistics.block_permutation_test(true=Y_true, pred=Y_pred, metric=correlation_score, 
			n_perms=N_PERMS, N_PROC=N_PROC)

		# save file out
		if p.session:
			permutations_fn = os.path.join(results_dir, f'{p.sub}_ses-{p.session}_task-{task}_desc-permutations_iter-{p.iteration}.npy')
		else:
			permutations_fn = os.path.join(results_dir, f'{p.sub}_task-{task}_desc-permutations_iter-{p.iteration}.npy')

		np.save(permutations_fn, permutations)

		print (f'Finished permutations for {task}')