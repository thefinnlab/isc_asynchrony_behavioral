import sys, os

sys.path.append('../../utils/') 

from config import *
import dataset_utils as utils
import nlp_utils as nlp
import encoding_utils as encoding
import statistics_utils as stats

import glob
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from operator import itemgetter
import torch

from nilearn.input_data import NiftiMasker
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend

from surfplot import Plot
from plotting_utils import vol_to_surf, make_layers_dict, plot_surf_data

from joblib import Parallel, delayed

N_PROC = 8
N_PERMS = 1000

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Dataset to use (and task name within datasets
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-test', '--test_tasks', type=str, nargs='+')
	parser.add_argument('-s', '--sub', type=str) 

	# type of analysis we're running --> linked to the name of the regressors
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-i', '--iteration', type=str)
	parser.add_argument('-f', '--features_list', nargs='+')
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	print (f'Fitting encoding model: features - {p.features_list}', flush=True)

	data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc-smooth/', p.sub)
	masks_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/masks/group/')
	results_dir = os.path.join(BASE_DIR, 'derivatives', 'results', p.dataset, p.sub, p.model_name)
	plots_dir = os.path.join(BASE_DIR, 'derivatives', 'plots', 'encoding_preds', p.dataset, p.sub, p.model_name)
	regressors_dir = os.path.join(BASE_DIR, 'derivatives/regressors/', p.dataset)

	utils.attempt_makedirs(results_dir)
	utils.attempt_makedirs(plots_dir)

	## some specific things for the dataset
	mask_fn = os.path.join(masks_dir, 'group-MNI152NLin6Asym_res-all_desc-brain_gm-mask-intersection.nii.gz')
	masker = NiftiMasker(mask_fn).fit()

	################################################
	#### Load and predict, then make distribution ##
	################################################

	# load model
	model_fn = os.path.join(results_dir, f'{p.sub}_encoding-model_iter-{p.iteration}.pkl')

	with open(model_fn, 'rb') as f:
		encoding_pipeline = joblib.load(f)

	for task in p.test_tasks:

		# load feature filenames
		if 'wheretheressmoke' in task:
			task_regressors_dir = os.path.join(regressors_dir, 'wheretheressmoke')
		else:
			task_regressors_dir = os.path.join(regressors_dir, task)

		all_feature_fns = []

		for feature in p.features_list:
			feature_fns = sorted(glob.glob(os.path.join(task_regressors_dir, feature, '*.npy')))
			all_feature_fns.append(feature_fns)

		# expand the list of lists
		# get all filenames and the grab their feature names
		all_feature_fns = sum(all_feature_fns, [])    
		feature_names = ['_'.join(Path(fn).stem.split('_')[1:]) for fn in all_feature_fns]
	
		print (f'List of features: {feature_names}', flush=True)

		# created banded features
		features, feature_space_info = encoding.load_banded_features(all_feature_fns, feature_names)
		
		## now load data
		if 'wheretheressmoke' in task:
			sub_data_fns = sorted(glob.glob(os.path.join(data_dir, f'*wheretheressmoke*hyperaligned.npy')))

			assert (len(sub_data_fns) == 5)
			ds_test = np.mean([np.load(fn) for fn in sub_data_fns], axis=0)
		else:
			sub_data_fn = glob.glob(os.path.join(data_dir, f'*{task}*hyperaligned.npy'))
			assert (len(sub_data_fn) == 1)
			ds_test = np.load(fn)

		# make the prediction
		Y_pred = encoding_pipeline.predict(features)
		Y_pred = np.asarray(Y_pred)

		# do some cleanup before tests
		del encoding_pipeline
		del features

		# run permutation test
		permutations = stats.block_permutation_test(true=ds_test, pred=Y_pred, metric=correlation_score, n_perms=N_PERMS, N_PROC=N_PROC)

		# save file out
		permutations_fn = os.path.join(results_dir, f'{p.sub}_task-{task}_desc-permutations_iter-{p.iteration}.npy')
		np.save(permutations_fn, permutations)

		print (f'Finished permutations for {task}')