import os, sys, glob
import pandas as pd
from scipy import stats

sys.path.append('../../utils/')

from config import *
from preproc_utils import match_df_distributions

def load_df_preproc(preproc_dir, stim):
	# read from src dir for raw comparison
	df_preproc = pd.read_csv(os.path.join(preproc_dir, f'{stim}/src/{stim}_transcript-preprocessed.csv'))
	df_preproc['task'] = stim
	return df_preproc

if __name__ == '__main__':

	ALPHA = 0.1
	N_ITER = 100
	TARGET_VARIABLE = 'Lg10WF'

	spoken_stimuli = ['wheretheressmoke', 'howtodraw', 'odetostepfather']
	written_stimuli = ['demon', 'keats']

	# set directories
	stim_dir = os.path.join(BASE_DIR, 'stimuli')
	gentle_dir = os.path.join(stim_dir, 'gentle')
	preproc_dir = os.path.join(stim_dir,'preprocessed')

	# load the spoken stimuli
	df_spoken = [load_df_preproc(preproc_dir, stim) for stim in spoken_stimuli]
	df_spoken = pd.concat(df_spoken).reset_index(drop=True)
	df_spoken['type'] = 'spoken'

	# trim to nwp words
	df_nwp_spoken = df_spoken[df_spoken['NWP_Candidate']].dropna()

	########################################################
	##### Match distribution of written stim to spoken #####
	########################################################

	df_written = []

	for stim in written_stimuli:

		df_preproc_fn = os.path.join(preproc_dir, stim,  f'{stim}_transcript-preprocessed')

		df_stim = load_df_preproc(preproc_dir, stim)
		df_nwp_stim = df_stim[df_stim['NWP_Candidate']].dropna()
	
		# make sure something is returned
		df_nwp_stim, t_stat, pval = match_df_distributions(df_nwp_stim, df_nwp_spoken, source_col=TARGET_VARIABLE, target_col=TARGET_VARIABLE, alpha=ALPHA, n_iter=N_ITER)

		# grab the frequency indices --> match them to the original dataframe
		frequency_indices = df_nwp_stim.index.to_numpy()
		frequency_filter = df_stim.index.isin(frequency_indices)

		# if it's in the frequency filter --> set the frequency filter, otherwise remove it as a possible candidate
		df_stim.loc[frequency_filter, 'Frequency_Filter'] = True
		df_stim.loc[~frequency_filter, ['NWP_Candidate', 'Frequency_Filter']] = False

		# only save to preproc dir, not to src dir in case need to edit
		df_stim.to_csv(f'{df_preproc_fn}.csv', index=False)
		df_stim.to_json(f'{df_preproc_fn}.json', orient='records')

		df_written.append(df_nwp_stim)

	############################################
	##### Ensure distributions are matched #####
	############################################

	df_nwp_written = pd.concat(df_written).reset_index(drop=True).dropna()
	df_nwp_written['type'] = 'written'

	result = stats.ttest_ind(df_nwp_spoken[TARGET_VARIABLE], df_nwp_written[TARGET_VARIABLE])

	print (f'Distribution difference (spoken v. written): t({result.df})={result.statistic:.2f}, p={result.pvalue:.2f}')