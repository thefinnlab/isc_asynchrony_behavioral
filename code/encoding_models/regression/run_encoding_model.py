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

sys.path.append('../../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import nlp, encoding, plotting

import torch

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Dataset to use (and task name within datasets
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-train', '--train_tasks', type=str, nargs='+')
	parser.add_argument('-test', '--test_tasks', type=str, nargs='+')
	parser.add_argument('-s', '--sub', type=str) 

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
	else:
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

	#############################################
	##### Load data (predictors / mri data) #####
	#############################################

	all_features = []
	all_data = []

	# combine tasks --> then split later
	all_tasks = p.train_tasks + p.test_tasks

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
	
		print (f'List of features: {feature_names}', flush=True)
	
		banded_info = encoding.load_banded_features(all_feature_fns, feature_names)
		all_features.append(banded_info) 

		print (banded_info[0].shape)

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

	#############################################
	#### Fit the model based on split features ###
	#############################################

	DELAYS = [1,2,3,4]
	N_ITER = 100

	if torch.cuda.is_available():
		print (f'Using torch_cuda backend!')
		backend = set_backend("torch_cuda", on_error="warn")
		N_JOBS = None
	else:
		N_JOBS = 8

	# we could use a loop over the train test split but here we'll just parallelize task
	train = [all_tasks.index(task) for task in p.train_tasks]
	test = [all_tasks.index(task) for task in p.test_tasks]

	# now build the pipeline
	_, encoding_pipeline = encoding.build_encoding_pipeline(
		X=itemgetter(*train)(features), 
		Y=itemgetter(*train)(all_data), 
		inner_folds='loo', 
		feature_space_infos=feature_space_info[0], 
		delays=DELAYS, 
		n_iter=N_ITER, 
		n_jobs=N_JOBS
	)

	# now save things out
	model_fn = os.path.join(results_dir, f'{p.sub}_encoding-model_iter-{p.iteration}.pkl')

	# Get train test data for this current split and fit the model
	X_train, Y_train, X_test, Y_test = encoding.get_train_test_splits(features, all_data, train, test)
	print (f'X train: {X_train.shape}, Y train: {Y_train.shape}, X test: {X_test.shape}, Y test: {Y_test.shape}', flush=True)

	encoding_pipeline.fit(X_train, Y_train)

	with open(model_fn, 'wb') as f:
		pipeline = _to_cpu(encoding_pipeline)
		pipeline = _to_numpy(pipeline)
		joblib.dump(pipeline, f)

	dss_test = []

	for task in p.test_tasks:

		# find the data for the current test task
		task_idx = all_tasks.index(task)
		_, _, X_test, Y_test = encoding.get_train_test_splits(features, all_data, train, [task_idx])

		# now predict the timeseries --> for now we only care about overall prediction
		Y_pred = encoding_pipeline.predict(X_test)
		prediction_corr = correlation_score(Y_test, Y_pred)

		if torch.cuda.is_available():
			Y_pred = tensor2numpy(Y_pred)
			prediction_corr = tensor2numpy(prediction_corr)

		# get the correlation
		residuals = (Y_test - Y_pred)

		residuals_fn = os.path.join(results_dir, f'{p.sub}_task-{task}_desc-residuals_iter-{p.iteration}.npy')
		pred_corr_fn = os.path.join(results_dir, f'{p.sub}_task-{task}_desc-prediction_iter-{p.iteration}.npy')
		
		np.save(pred_corr_fn, prediction_corr)
		np.save(residuals_fn, residuals)

		dss_test.append(prediction_corr)

	dss_test = np.vstack(dss_test)
	max_val = np.max([np.nanmax(abs(ds)) for ds in dss_test])

	for task, ds in zip(p.test_tasks, dss_test):
		# now plot the predictions
		vol_corr = masker.inverse_transform(ds)

		# plot the volume correlation
		surfs, data = vol_to_surf(vol_corr, surf_type='fsaverage', map_type='inflated')
		layer = make_layers_dict(data=data, cmap='RdBu_r', label=f'Prediction (r)', alpha=1, color_range=(-max_val, max_val))

		plot_fn = os.path.join(plots_dir, f'{p.sub}_task-{task}_desc-prediction_iter-{p.iteration}.{EXT}')
		title = f'{p.sub} - {task} encoding prediction, {p.model_name} iter - {p.iteration}'

		if PLOT_SEPARATE_VIEWS:
			for view in VIEWS:
				_ = plot_surf_data(surfs, [layer], views=[view], colorbar=COLORBAR, surf_type=SURF_TYPE, 
					add_depth=ADD_DEPTH, out_fn=os.path.join(plots_dir, plot_fn.replace(f'.{EXT}', f'-{view}.{EXT}')),
					title=title)
		else:
			_ = plot_surf_data(surfs, [layer], views=VIEWS, colorbar=COLORBAR, surf_type=SURF_TYPE, 
				add_depth=ADD_DEPTH, out_fn=os.path.join(plots_dir, plot_fn), title=title)