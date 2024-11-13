from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.distance import cdist

def get_time_segments(n_items, window_size):

	# total number of segments is number of items, accounting for the window size --> add 1 for inclusive
	n_segments = n_items - window_size + 1
	time_segments = np.asarray([np.arange(i, i + window_size) for i in range(0, n_segments)])

	return time_segments

def generate_foil_segments(time_segments, current_segment, window_size, buffer_size):
	
	#generate an arrays to buffer around the start and end times
	start_buffer = np.arange(-buffer_size + current_segment[0], current_segment[0])
	end_buffer = np.arange(current_segment[-1] + 1, current_segment[-1] + buffer_size + 1)
	
	# get current segment with buffer of timepoints
	segment = np.hstack((start_buffer, current_segment, end_buffer))

	# find the indices that don't have overlap with the current timewindow (segment + buffer) 
	foil_idxs = np.where(~np.isin(time_segments, segment).any(axis=1))[0]
	
	return time_segments[foil_idxs]

def timepoint_classification(ds_train, ds_test, window_size, buffer_size, metric='correlation'):

	#ensure the two datasets are equal length
	assert ds_test.shape[0] == ds_train.shape[0]

	n_items = ds_train.shape[0]
	time_segments = get_time_segments(n_items, window_size)
	
	results = []

	for i, segment in enumerate(time_segments):
	
		# get indices for all the train data
		foil_idxs = generate_foil_segments(time_segments, segment, window_size, buffer_size)
	
		train = np.concatenate([
			ds_train[segment][np.newaxis], # add the true comparison segment to start of array
			ds_train[foil_idxs] # foil segments
		])
	
		# flatten all features ==> now is n_segments x (samples * features)
		train = train.reshape(train.shape[0], -1)
		test = ds_test[segment].flatten()[np.newaxis]
	
		# find distance between train and test segments
		distances = cdist(train, test, metric=metric)
	
		# accuracy --> first item is ground truth, therefore we score based on if 
		# that was accurately classified
		accuracy = np.argmin(distances) == 0 
		results.append((i, accuracy)) # keep track index and accuracy
		
	# make sure the results are ordered by timepoints
	idxs, accuracy = np.stack(results).T
	accuracy = accuracy[np.argsort(idxs)]
		
	return accuracy

def run_searchlight_timepoint_classification(searchlights, ds_train, ds_test, window_size, buffer_size, metric='correlation', nproc=1):
	
	#for each node's searchlight, create a job to run it in parallel
	jobs = []
	
	for sl in searchlights:
		_train = ds_train[:, sl]
		_test = ds_test[:, sl]
		job = delayed(timepoint_classification)(_train, _test, window_size, buffer_size, metric)
		jobs.append(job)
	
	# run the sl jobs in parallel, aggregate into accuracy
	with Parallel(n_jobs=nproc) as parallel:
		accuracy = parallel(jobs)
		
	accuracy = np.array(accuracy)
	
	return accuracy