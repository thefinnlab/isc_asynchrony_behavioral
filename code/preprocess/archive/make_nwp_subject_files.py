import sys, os
import numpy as np
import pandas as pd
from collections import Counter

sys.path.append('/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavioral/code/utils/')

from randomization_utils import create_balanced_orders, get_consecutive_list_idxs, sort_consecutive_constraint

if __name__ == "__main__":

	EXPERIMENT_NAME = 'pilot-version-04'
	TASK = 'black'

	percent_sampled = 0.25 # number of items to sample for each subject
	n_counts_per_item = 25 # number of times items are seen across subjects

	# set directories
	base_dir = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavioral/'
	preproc_dir = os.path.join(base_dir, 'stimuli', 'preprocessed')
	task_out_dir = os.path.join(base_dir, 'stimuli', 'presentation_orders', EXPERIMENT_NAME, TASK)

	if not os.path.exists(task_out_dir):
		os.makedirs(task_out_dir)

	# load preprocessed transcript
	df_task_preproc_fn = os.path.join(preproc_dir, TASK, f'{TASK}_transcript_preprocessed')
	df_preproc = pd.read_csv(f'{df_task_preproc_fn}.csv')

	# find indices for presentation and set number of items each subject sees
	nwp_indices = np.where(df_preproc['NWP_Candidate'])[0]
	n_items_per_subject = round(len(nwp_indices) * percent_sampled)

	# create experiment structure for subjects --> sort the indices
	subject_experiment_orders = create_balanced_orders(items=nwp_indices, n_elements_per_subject=n_items_per_subject, use_each_times=n_counts_per_item)
	subject_experiment_orders = list(map(sorted, subject_experiment_orders))

	# Find lists with consecutive items violating our constraint
	idxs = get_consecutive_list_idxs(subject_experiment_orders, consecutive_length=2)
	subject_experiment_orders = sort_consecutive_constraint(subject_experiment_orders, consecutive_length=2)

	# Test again once we have completed resorting
	idxs = get_consecutive_list_idxs(subject_experiment_orders, consecutive_length=2)
	print (f'Lists violating consecutive index constraint: {100*(len(idxs))/len(subject_experiment_orders)}%')

	uniq, counts = np.unique(subject_experiment_orders, return_counts=True)
	print (f'All counts per word: {np.sum(counts >= n_counts_per_item) / len(counts)*100}%')

	counts = Counter(tuple(o) for o in subject_experiment_orders)
	unique_orders = np.sum([v for k, v in counts.items()]) / len(counts)

	print (f'Unique orders: {unique_orders*100}%')

	orders_meeting_consecutive = np.sum([(np.all(np.diff(order) >= 2)) for order in subject_experiment_orders]) / len(subject_experiment_orders)
	print (f'Consecutive constraint: {orders_meeting_consecutive*100}%')

	for i, order in enumerate(subject_experiment_orders):
		# find indices not selected for the current subject and set to false
		df_subject = df_preproc.copy()
		unselected = np.setdiff1d(nwp_indices, order)
		df_subject['NWP_Candidate'].loc[unselected] = False

		sub_fn = os.path.join(task_out_dir, f'sub-{str(i+1).zfill(5)}_task-{TASK}.json')
		df_subject.to_json(sub_fn, orient='records')
