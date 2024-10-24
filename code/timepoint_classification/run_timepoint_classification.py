'''
Timepoint classification on fsaverage5 surface
'''
import sys, os
import glob
import argparse
import numpy as np
import pandas as pd 

from scipy.stats import zscore
import searchlights as sls

sys.path.append('../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import statistics

from timepoint_classification import run_searchlight_timepoint_classification

def load_subject_data(data_dir, sub_list, task, session):
	'''
	Load data for given subjects --> zscore and return an 
	array of size (sub x trs x nodes)
	'''

	dss = []

	for sub in sub_list:
		sub_data_fns = glob.glob(os.path.join(data_dir, sub, f'*{session}*{task}*hyperaligned.npy'))
		
		assert (len(sub_data_fns) == 1)
		
		ds_sub = zscore(np.load(sub_data_fns[0]), axis=0)
		dss.append(ds_sub)
	
	return np.stack(dss)

def masked_to_fsaverage(mask, results):

	# get trs x n_nodes
	n_trs = results.shape[0]
	n_nodes = mask.shape[0]

	# create an empty array and map the results to that array
	arr = np.zeros((n_trs, n_nodes))
	arr[:, mask] = results
	
	return arr

HEMIS = ['l', 'r']
SURF = 'fsaverage5'

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Dataset to use (and task name within datasets
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-s', '--sub', type=str) 
	parser.add_argument('-t', '--task', type=str)

	## session information
	parser.add_argument('-train_session', '--train_session', type=str) 
	parser.add_argument('-test_session', '--test_session', type=str) 

	# timepoint classification information
	parser.add_argument('-radius', '--radius', type=int, default=20)
	parser.add_argument('-window_size', '--window_size', type=int, default=5) 
	parser.add_argument('-buffer_size', '--buffer_size', type=int, default=5) 
	parser.add_argument('-n_proc', '--n_proc', type=int, default=1) 

	p = parser.parse_args()

	print (f'Timepoint classification: {p.sub}', flush=True)
	print (f'Task: {p.task}', flush=True)
	print (f'Train session: {p.train_session}', flush=True)
	print (f'Test session: {p.test_session}', flush=True)

	if p.dataset == 'deniz-readinglistening':
		# get data dir
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc/')

		# get subjects excluding their html files
		sub_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub*[!html]')))
		sub_list = [os.path.basename(d) for d in sub_dirs]

		# remove the current subject from the group
		sub_list = set(sub_list).difference([p.sub])

	results_dir = os.path.join(BASE_DIR, 'derivatives/results/', p.dataset, p.sub, 'timepoint_classification')
	utils.attempt_makedirs(results_dir)

	#######################################
	##### Load data for subject/group #####
	#######################################

	# load data for the subject and the rest of the group
	ds_sub = load_subject_data(data_dir, [p.sub], task=p.task, session=p.train_session)
	ds_group = load_subject_data(data_dir, sub_list, task=p.task, session=p.test_session)

	# average across subjects (either all subjects in the group, or the subject to themselves)
	ds_sub = np.nanmean(ds_sub, axis=0)
	ds_group = np.nanmean(ds_group, axis=0)

	####################################################
	##### Run searchlight timepoint classification #####
	####################################################

	ds_results = []

	for i, hemi in enumerate(HEMIS):
		# load mask and searchlights for current hemisphere
		surf_mask = sls.get_mask(hemi, SURF)
		searchlights = sls.get_searchlights(hemi, p.radius, SURF)

		# get the current hemisphere for both subject and group
		ds_sub_hemi = np.split(ds_sub, 2, axis=1)[i]
		ds_group_hemi = np.split(ds_group, 2, axis=1)[i]

		# mask the subject and groups
		ds_sub_hemi = ds_sub_hemi[:, surf_mask]
		ds_group_hemi = ds_group_hemi[:, surf_mask]

		# run timepoint classification
		# should be invariant to train-test direction (since what we're doing a correlation)
		results = run_searchlight_timepoint_classification(
			searchlights=searchlights, 
			ds_train=ds_sub_hemi,
			ds_test=ds_group_hemi,
			window_size=p.window_size,
			buffer_size=p.buffer_size,
			nproc=p.n_proc,
		)

		# results is of size (n_nodes x timepoints) 
		# put time as first dimension and map back to fsaverage
		results = masked_to_fsaverage(surf_mask, results.T)
		ds_results.append(results)

	# stack the hemispheres together
	ds_results = np.concatenate(ds_results, axis=1)

	#######################################
	##### Save classification results #####
	#######################################

	out_fn = os.path.join(results_dir, f'{p.sub}_task-{p.task}_train-{p.train_session}_test-{p.test_session}_timepoint-classification.npy')
	np.save(out_fn, ds_results)