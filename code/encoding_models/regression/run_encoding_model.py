import sys, os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from operator import itemgetter
from sklearn.model_selection import KFold

from nilearn.input_data import NiftiMasker
from himalaya.backend import set_backend

from scipy import sparse

sys.path.append('../../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import nlp, encoding, plotting

import torch

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Dataset to use (and task name within datasets
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-s', '--sub', type=str) 

	parser.add_argument('-train', '--train_tasks', type=str, nargs='+')
	parser.add_argument('-test', '--test_tasks', type=str, nargs='+')
	parser.add_argument('-ses', '--session', type=str, default=None) # session is mostly used to stratify tasks

	
	# type of analysis we're running --> linked to the name of the regressors
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-i', '--iteration', type=str)
	parser.add_argument('-f', '--features_list', nargs='+')
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	print (f'Fitting encoding model for {p.sub}', flush=True)
	print (f'Train tasks: {p.train_tasks}', flush=True)
	print (f'Test tasks: {p.test_tasks}', flush=True)
	print (f'Features: {p.features_list}', flush=True)

	if p.dataset == 'deniz-readinglistening':
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc/', p.sub)
		# masks_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc/', p.sub, 'mappers')

		# load the subject mapping to fsaverage
		# mapper_fn = glob.glob(os.path.join(masks_dir, '*fsaverage*'))
		# assert (len(mapper_fn) == 1)

		# mapper = sparse.load_npz(mapper_fn[0])
	else:
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc-smooth/', p.sub)
		masks_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/masks/group/')

		## some specific things for the dataset
		mask_fn = os.path.join(masks_dir, 'group-MNI152NLin6Asym_res-all_desc-brain_gm-mask-intersection.nii.gz')
		masker = NiftiMasker(mask_fn).fit()

	results_dir = os.path.join(BASE_DIR, 'derivatives/results/', p.dataset, p.sub, p.model_name)
	plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/encoding_preds/', p.dataset, p.sub, p.model_name)
	regressors_dir = os.path.join(BASE_DIR, 'derivatives/regressors/', p.dataset)

	utils.attempt_makedirs(results_dir)
	utils.attempt_makedirs(plots_dir)

	#############################################
	##### Load data (predictors / mri data) #####
	#############################################

	all_features = []
	all_data = []

	# combine tasks --> then split later
	all_tasks = p.train_tasks + p.test_tasks
	all_tasks = sorted(set(all_tasks))

	# load features for each task
	for task in all_tasks:

		all_feature_fns = []

		if 'wheretheressmoke' in task:
			task_regressors_dir = os.path.join(regressors_dir, 'wheretheressmoke')
		else:
			task_regressors_dir = os.path.join(regressors_dir, task)

		for feature in p.features_list:
			feature_fns = sorted(glob.glob(os.path.join(task_regressors_dir, feature, '*.npy')))
			all_feature_fns.append(feature_fns)

		# expand the list of lists
		# get all filenames and the grab their feature names
		all_feature_fns = sum(all_feature_fns, [])    
		feature_names = ['_'.join(Path(fn).stem.split('_')[1:]) for fn in all_feature_fns]

		print (f'Loading features for task: {task}', flush=True)
		print (f'List of features: {feature_names}', flush=True)
	
		banded_info = encoding.load_banded_features(all_feature_fns, feature_names)
		all_features.append(banded_info) 

		print (banded_info[0].shape)

		if p.dataset == 'deniz-readinglistening':
			if p.session:
				sub_data_fn = glob.glob(os.path.join(data_dir, f'*{p.session}*{task}*hyperaligned.npy'))
			else:
				sub_data_fn = glob.glob(os.path.join(data_dir, f'*{task}*hyperaligned.npy'))
		else:
			sub_data_fn = glob.glob(os.path.join(data_dir, f'*{task}*hyperaligned.npy'))

		assert (len(sub_data_fn) == 1)

		ds = np.load(sub_data_fn[0])

		print (ds.shape)

		# load the mri data
		all_data.append(ds)

	# expand into features and the info
	features, feature_space_info = zip(*all_features)

	# make sure dimensions of features are equal across the feature spaces
	assert (utils.all_equal(feature_space_info))

	###############################################
	#### Fit the model based on split features ####
	###############################################

	# set the rest of the parameters
	N_ITER = 400
	N_ALPHAS_BATCH = None
	N_TARGETS_BATCH = 1000
	N_TARGETS_BATCH_REFIT = 1000

	# TR is 2s so delay at 2,4,6,8s
	DELAYS = [1,2,3,4]

	## Determine if we want GPU support
	if torch.cuda.is_available():
		print (f'Using torch_cuda backend!')
		backend = set_backend("torch_cuda", on_error="warn")
		N_JOBS = None
	else:
		N_JOBS = -1

	## Find the train and test tasks in the indices
	train = [all_tasks.index(task) for task in p.train_tasks]
	test = [all_tasks.index(task) for task in p.test_tasks]

	# Set up cross-validation to use LOO CV
	# Divide our features into our training data
	X_train = itemgetter(*train)(features)
	n_samples = np.concatenate(X_train).shape[0]
	run_lengths = [len(x) for x in X_train]
	run_onsets = np.cumsum(np.concatenate(([0], run_lengths)))[:-1]

	print (f'Number of samples: {n_samples}')
	print (f'Run lengths: {run_lengths}')

	inner_cv = encoding.generate_leave_one_run_out(n_samples, run_onsets, n_runs_out=1) #, train_mean=True)
	inner_cv = encoding.check_cv(inner_cv)  # copy the cross-validation splitter into a reusable list

	## Now set up the model
	encoding_pipeline = encoding.build_encoding_pipeline(
		X=features, 
		Y=all_data, 
		inner_cv=inner_cv,
		feature_space_infos=feature_space_info[0], 
		solver="random_search",
		n_alphas_batch=N_ALPHAS_BATCH,
		n_targets_batch=N_TARGETS_BATCH,
		n_targets_batch_refit=N_TARGETS_BATCH_REFIT,
		delays=DELAYS, 
		n_iter=N_ITER, 
		n_jobs=N_JOBS,
	)

	## Get train test data for this current split and fit the model
	X_train, Y_train, X_test, Y_test = encoding.get_train_test_splits(features, all_data, train, test)
	print (f'X train: {X_train.shape}, Y train: {Y_train.shape}, X test: {X_test.shape}, Y test: {Y_test.shape}', flush=True)

	encoding_pipeline.fit(X_train, Y_train)

	#############################################
	###### Make predictions from the model ######
	#############################################

	dss_test = []

	# load features for each space
	behavioral_feature_type = ['ground-truth', 'model-predicted', 'human-audio', 'human-text']

	for task in p.test_tasks:

		if 'wheretheressmoke' in task:
			task_regressors_dir = os.path.join(regressors_dir, 'wheretheressmoke')
		else:
			task_regressors_dir = os.path.join(regressors_dir, task)

		for behavioral_type in behavioral_feature_type:

			# find the data for the current task
			task_idx = all_tasks.index(task)
			_, _, X_test, Y_test = encoding.get_train_test_splits(features, all_data, train, [task_idx])

			# if these are not ground truth features, we load behavioral features and replace X_test
			if behavioral_type != 'ground-truth':

				# load the model features
				all_feature_fns = []

				# base features are everything besides the current model
				base_features = sorted(set(p.features_list).difference([p.model_name]))

				for feature in base_features:
					feature_fns = sorted(glob.glob(os.path.join(task_regressors_dir, feature, '*.npy')))
					all_feature_fns.append(feature_fns)

				# now get the behavioral features
				behavioral_feature_fns = sorted(glob.glob(os.path.join(task_regressors_dir,  f'behavioral/{p.model_name}/*{behavioral_type}*.npy')))

				# combine base features with features from the behavior paradigm
				all_feature_fns = sum([*all_feature_fns, behavioral_feature_fns], [])
				all_feature_names = ['_'.join(Path(fn).stem.split('_')[1:]) for fn in all_feature_fns]
		
				# load the features
				behavioral_features, behavioral_feature_names = encoding.load_banded_features(all_feature_fns, all_feature_names)

				print (f'Loaded features for {behavioral_type} {p.model_name}')

				X_test = behavioral_features		

			# get all metrics of interest (for now correlation, r2, prediction, and residuals)
			results = encoding.get_all_banded_metrics(encoding_pipeline, X_test, Y_test)

			# save all the metrics
			results_fns = {}

			for metric, data in results.items():

				# include session as a part of the name
				if p.session:
					metric_fn = os.path.join(results_dir, f'{p.sub}_ses-{p.session}_task-{task}_desc-{metric}_dtype-{behavioral_type}_iter-{p.iteration}.npy')
				else:
					metric_fn = os.path.join(results_dir, f'{p.sub}_task-{task}_desc-{metric}_dtype-{behavioral_type}_iter-{p.iteration}.npy')

				np.save(metric_fn, data)

				results_fns[metric] = metric_fn

			# # map to fsaverage so that we can plot
			# if p.dataset == 'deniz-readinglistening':
			# 	results = {k: data @ mapper.T for k, data in results.items()}

			dss_test.append((task, behavioral_type, 'r2', results['r2']))
			dss_test.append((task, behavioral_type, 'correlation', results['correlation']))

	#############################################
	###### Plot predictions from the model ######
	#############################################

	for task, behavioral_type, metric, ds in dss_test:

		max_val = np.nanmax(abs(ds))

		if p.dataset == 'deniz-readinglistening':
			surfs, data = plotting.numpy_to_surface(ds)
		else:
			vol_corr = masker.inverse_transform(ds)
			surfs, data = plotting.vol_to_surf(vol_corr, surf_type='fsaverage', map_type='inflated')

		layer = plotting.make_layers_dict(data=data, cmap='RdBu_r', label=f'{metric}', alpha=1, color_range=(-max_val, max_val))

		if p.session:
			plot_fn = os.path.join(plots_dir, f'{p.sub}_ses-{p.session}_task-{task}_desc-{metric}_dtype-{behavioral_type}.{EXT}')
			title = f'{p.sub} - {p.session} {task} encoding {metric}, {p.model_name} {behavioral_type}'
		else:
			plot_fn = os.path.join(plots_dir, f'{p.sub}_task-{task}_desc-{metric}_dtype-{behavioral_type}.{EXT}')
			title = f'{p.sub} - {task} encoding {metric}, {p.model_name} {behavioral_type}'

		if PLOT_SEPARATE_VIEWS:
			for view in VIEWS:
				_ = plotting.plot_surf_data(surfs, [layer], views=[view], colorbar=COLORBAR, surf_type=SURF_TYPE, 
					add_depth=ADD_DEPTH, out_fn=os.path.join(plots_dir, plot_fn.replace(f'.{EXT}', f'-{view}.{EXT}')),
					title=title)
		else:
			_ = plotting.plot_surf_data(surfs, [layer], views=VIEWS, colorbar=COLORBAR, surf_type=SURF_TYPE, 
				add_depth=ADD_DEPTH, out_fn=os.path.join(plots_dir, plot_fn), title=title)