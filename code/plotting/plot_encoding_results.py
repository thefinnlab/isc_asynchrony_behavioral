import sys, os
import glob
import numpy as np
import pandas as pd 
import argparse
import nibabel as nib
from nilearn.input_data import NiftiMasker
from itertools import product

from matplotlib import pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

from himalaya.scoring import correlation_score
from scipy import sparse
from neuromaps.transforms import fsaverage_to_fsaverage

sys.path.append('../utils/') 

from config import *
import dataset_utils as utils
from tommy_utils import statistics, plotting, encoding

def ztransform_mean(dss):
	dss = np.stack(dss)
	return np.tanh(np.mean([np.arctanh(ds) for ds in dss], axis=0))

def calculate_error(Y_true, Y_pred, crit_trs):
	absolute_error = abs(Y_true - Y_pred)[crit_trs, :]

	mse = (Y_true - Y_pred)**2
	mse = mse[crit_trs, :]
	
	return absolute_error, mse

def conduct_feature_contrasts(ds_dict, feature_contrasts):

	all_contrasts = {}
	
	for contrast in feature_contrasts:
		ds_a, ds_b = [ds_dict[c] for c in contrast]

		contrast_key = ' - '.join(contrast)

		all_contrasts[contrast_key] = ds_a - ds_b
		print (f'Finished contrast {contrast}')

	return all_contrasts

def create_spoken_written_cmap(continuous=True):

	if continuous:
		spoken_written_cmap = clr.LinearSegmentedColormap.from_list('spoken-written', ['#005208', '#72D16B', '#F7F6F6', '#E4B266', '#623800'], N=256)
		spoken_written_cmap = spoken_written_cmap.reversed()
	else:
		spoken_written_cmap = sns.color_palette('BuPu', n_colors=9)
		spoken_written_cmap.insert(0, '#82C564')
		spoken_written_cmap.insert(1, '#F7CD84')
	return spoken_written_cmap


N_SUBS_SIGNIF_THRESHOLD = 3

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-ses', '--session', type=str, default='')
	parser.add_argument('-m', '--model_name', type=str)
	p = parser.parse_args()

	results_dir = os.path.join(BASE_DIR, 'derivatives/results/', p.dataset) #, p.sub, p.model_name)
	plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/encoding_preds/', p.dataset, 'group')

	utils.attempt_makedirs(plots_dir)

	if p.dataset == 'deniz-readinglistening':
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc/')
	else:
		data_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/dark-matter-preproc-smooth/')
		masks_dir = os.path.join(DATASETS_DIR, p.dataset, 'derivatives/masks/group/')
		mask_fn = os.path.join(masks_dir, 'group-MNI152NLin6Asym_res-all_desc-brain_gm-mask-intersection.nii.gz')
		masker = NiftiMasker(mask_fn).fit()

	# get subjects excluding their html files
	sub_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub*[!html]')))
	sub_list = [os.path.basename(d) for d in sub_dirs]

	#########################################
	####### Get significance counts #########
	#########################################

	# go through each subject and load their permutations
	dss = []

	for sub in sub_list:

		sub_data_dir = os.path.join(data_dir, sub)
		sub_results_dir = os.path.join(results_dir, sub, p.model_name)

		# set filenames
		sub_data_fn = sorted(glob.glob(os.path.join(sub_data_dir, f'*{p.session}*{p.task}*hyperaligned.npy')))
		sub_results_fn = sorted(glob.glob(os.path.join(sub_results_dir, f'*{p.session}*{p.task}*prediction*ground-truth*.npy')))
		sub_permutations_fn = glob.glob(os.path.join(sub_results_dir, f'*{p.session}*{p.task}*permutations*.npy'))

		# load ground-truth predictions --> predictions are the same regardless of run since weights dont change
		if p.task == 'wheretheressmoke':
			assert (len(sub_permutations_fn) == 1 and (len(sub_data_fn) == 5 or len(sub_data_fn) == 2))
		else:
			assert (len(sub_permutations_fn) == 1 and len(sub_data_fn) == 1)

		# load results, average and correlate with ground truth
		ds_test = np.mean([np.load(fn) for fn in sub_data_fn], axis=0)
		Y_pred = np.mean([np.load(fn) for fn in sub_results_fn], axis=0)
		
		# load the data
		distribution = np.load(sub_permutations_fn[0])
		results = correlation_score(ds_test, Y_pred)
		
		# compare the results to the distribution and then find what is significant
		zvals, pvals = statistics.p_from_null(results[np.newaxis], distribution, mult_comp_method='fdr_bh', axis=0)    
		thresholded = statistics.pvalue_threshold(results, pvals)
		significant_voxels = ~np.isnan(thresholded)

		dss.append([significant_voxels, results, pvals])

	# stack the results together
	sig_voxels, pred_accuracy, pvals = [np.stack(ds) for ds in zip(*dss)]
	avg_accuracy = ztransform_mean(pred_accuracy)

	# now count number of participants 
	sig_counts = np.sum(sig_voxels, axis=0).astype(float)
	sig_counts[sig_counts < 1] = np.nan

	#########################################
	####### Plot significance counts ########
	#########################################

	# set title and filename for writing
	if p.session:
		ses = f'_ses-{p.session}_'
	else:
		ses = '_'
	
	out_fn = os.path.join(plots_dir, f'group{p.session}task-{p.task}_nsubs-significance.{EXT}')

	if TITLE:
		title = f'{p.dataset} - {p.task} {p.session}, significance map'
	else:
		title = '' #f'{dataset} - All {session} stories, n_subs significance'#f'group - n_subs significance'

	significance_cmap = plt.cm.get_cmap('magma', len(sub_list))    # 11 discrete colorsp
	max_val = np.nanmax(sig_counts)

	if p.dataset == 'deniz-readinglistening':
		surfs, data = plotting.numpy_to_surface(sig_counts, target_density='41k', method='nearest')
	else:
		ds_signif = masker.inverse_transform(sig_counts)
		surfs, data = plotting.vol_to_surf(ds_signif, surf_type='fsaverage', map_type='inflated', method='nearest')

	layer = plotting.make_layers_dict(data=data, cmap=significance_cmap, label=f'N_Subjects Significant', alpha=1, color_range=(1, len(sub_list)))

	if PLOT_SEPARATE_VIEWS:
		for view in VIEWS:
			_ = plotting.plot_surf_data(surfs, [layer], views=[view], 
				colorbar=COLORBAR if view == 'medial' else False, 
				surf_type=SURF_TYPE, 
				add_depth=ADD_DEPTH, out_fn=out_fn.replace(f'.{EXT}', f'-{view}.{EXT}'),
				title=title)
	else:
		_ = plotting.plot_surf_data(surfs, [layer], views=VIEWS, colorbar=COLORBAR, surf_type=SURF_TYPE, 
			add_depth=ADD_DEPTH, out_fn=out_fn, title=title)

	#########################################
	####### Determine critical word Trs #####
	#########################################

	# load gentle trasncript
	gentle_dir = os.path.join(DATASETS_DIR, p.dataset, 'stimuli', 'gentle', p.task)
	transcript_fn = os.path.join(gentle_dir, 'align.json')

	df_transcript = encoding.load_gentle_transcript(
		transcript_fn=transcript_fn, 
		start_offset=None #stim_times[0] if stim_times else None
	)

	## MAKE ADJUSTMENTS TO MODEL FEATURES FROM BEHAVIOR ########
	behavior_results_fn = os.path.join(BASE_DIR, f'derivatives/results/behavioral/task-{p.task}_group-analyzed-behavior_human-model-lemmatized.csv')
	behavior_results = pd.read_csv(behavior_results_fn)

	crit_idxs = np.unique(behavior_results['word_index'])

	# get critical word trs
	critical_word_trs = df_transcript.loc[crit_idxs, ['start', 'end']] // 2
	critical_word_trs = critical_word_trs['end'].astype(int).to_numpy()
	print (f'Total trs: {len(critical_word_trs)}')

	# reduce to unique critical words
	critical_word_trs = np.unique(critical_word_trs)
	print (f'Unique trs: {len(critical_word_trs)}')

	#########################################
	####### Conduct contrasts of MSE ########
	#########################################

	feature_contrasts = [
		['model-predicted', 'human-audio'],
		['model-predicted', 'human-text'],
		# ['model-predicted', 'ground-truth'],
		# ['human-audio', 'ground-truth'],
		# ['human-text', 'ground-truth'],
		['human-audio', 'human-text'] 
	]

	behavioral_feature_types = ['model-predicted', 'human-audio', 'human-text'] #'ground-truth', 

	# initialize dictionary of contrasts
	all_contrasts = {' - '.join(contrast): [] for contrast in feature_contrasts}
	all_sub_residuals = {dtype: [] for dtype in behavioral_feature_types}

	for sub, pvals in zip(sub_list, pvals):
		
		sub_results_dir = os.path.join(results_dir, sub, p.model_name)
		sub_data_dir = os.path.join(data_dir, sub)
		
		ds_sub_residuals = {} 

		for behavioral_type in behavioral_feature_types:

			sub_data_fn = sorted(glob.glob(os.path.join(sub_data_dir, f'*{p.session}*{p.task}*hyperaligned.npy')))
			sub_results_fn = sorted(glob.glob(os.path.join(sub_results_dir, f'*{p.session}*{p.task}*prediction*{behavioral_type}*.npy')))

			# load ground-truth predictions --> predictions are the same regardless of run since weights dont change
			if p.task == 'wheretheressmoke':
				assert (len(sub_data_fn) == 5 or len(sub_data_fn) == 2)
			else:
				assert (len(sub_data_fn) == 1)

			# load results, average and correlate with ground truth
			Y_true = np.mean([np.load(fn) for fn in sub_data_fn], axis=0)
			Y_pred = np.mean([np.load(fn) for fn in sub_results_fn], axis=0)
		
			absolute_error, mse = calculate_error(Y_true, Y_pred, critical_word_trs)
			ds_sub_residuals[behavioral_type] = mse

			print (f'Finished {sub} {behavioral_type}')

		# df_sub_residuals = pd.concat(df_sub_residuals)
		sub_contrast_dict = conduct_feature_contrasts(ds_sub_residuals, feature_contrasts)
		
		for k, v in all_contrasts.items():
			v.append(sub_contrast_dict[k])

		for k, v in all_sub_residuals.items():
			v.append(ds_sub_residuals[k])

	all_contrasts = {k: np.stack(v).squeeze() for k, v in all_contrasts.items()}
	all_sub_residuals = {k: np.stack(v).squeeze() for k, v in all_sub_residuals.items()}

	#########################################
	######## Plot accuracy/contrasts ########
	#########################################

	# accuracy contains both stat test thresholded and unthresholded accuracy
	thresholds = ['thresholded', 'unthresholded']
	signif_threshold = np.logical_or(np.isnan(sig_counts), sig_counts < N_SUBS_SIGNIF_THRESHOLD)

	# calculate humans > model (regardless of condition)
	human_v_model = (all_contrasts['model-predicted - human-audio'] + all_contrasts['model-predicted - human-text']) / 2
	spoken_v_written = -1 * all_contrasts['human-audio - human-text']

	spoken_written_cmap = create_spoken_written_cmap()

	plot_contrasts = [
		(avg_accuracy, 'prediction-accuracy', 'Prediction (r)', 'RdBu_r'),
		(human_v_model, 'model-human-contrast', 'Model - Human', 'RdBu_r'),
		(spoken_v_written, 'spoken-written-contrast', 'Spoken - Written', spoken_written_cmap),
	]

	for threshold, (ds_contrast, ptype, label, cmap) in product(thresholds, plot_contrasts):

		if ptype != 'prediction-accuracy':
			# average across subjects then trs
			ds_contrast = np.nanmean(np.nanmean(ds_contrast, axis=0), axis=0) #.mean(1).mean(0)
	
		if threshold == 'thresholded':
			ds = ds_contrast.copy()
			ds[signif_threshold] = np.nan
		else:
			ds = ds_contrast.copy()
		
		out_fn = os.path.join(plots_dir, f'group{ses}task-{p.task}_{ptype}-{threshold}.{EXT}')

		if TITLE:
			title = f'{p.dataset} - {p.task} {p.session}, {ptype}'
		else:
			title = '' #f'{dataset} - All {session} stories, n_subs significance'#f'group - n_subs significance'
		max_val = np.nanmax(abs(ds))
		
		if p.dataset == 'deniz-readinglistening':
			surfs, data = plotting.numpy_to_surface(ds, target_density='41k')
		else:
			ds = masker.inverse_transform(ds)
			surfs, data = plotting.vol_to_surf(ds, surf_type='fsaverage', map_type='inflated')
		
		layer = plotting.make_layers_dict(data=data, cmap=cmap, label=label, alpha=1, color_range=(-max_val, max_val))
	
		if PLOT_SEPARATE_VIEWS:
				for view in VIEWS:
					_ = plotting.plot_surf_data(surfs, [layer], views=[view], 
						colorbar=COLORBAR if view == 'medial' else False, 
						surf_type=SURF_TYPE, 
						add_depth=ADD_DEPTH, out_fn=out_fn.replace(f'.{EXT}', f'-{view}.{EXT}'),
						title=title)
		else:
			_ = plotting.plot_surf_data(surfs, [layer], views=VIEWS, colorbar=COLORBAR, surf_type=SURF_TYPE, 
				add_depth=ADD_DEPTH, out_fn=out_fn, title=title)