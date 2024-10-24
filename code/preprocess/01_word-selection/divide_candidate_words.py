import os, sys, glob
import json
from operator import itemgetter
import re
import numpy as np
import pandas as pd
import shutil
import argparse
from praatio import textgrid as tgio

sys.path.append('../../utils/')

from config import *
from preproc_utils import load_model_results, divide_nwp_dataframe, select_prediction_words

# FOR DIVIDING THE MODEL RESULTS INTO QUADRANTS
ACCURACY_TYPE = 'fasttext_avg_accuracy'
ACCURACY_PERCENTILE = 45
WINDOW_SIZE = 100
TOP_N = 5

# FOR FILTERING AND SELECTING WORDS FROM QUADRANTS
REMOVE_PERC = 0.5
SELECT_PERC = 0.45 # 0.39 for howtodraw, 0.45 for odetostepfather, #0.45 demon
MIN_SPACING_THRESH = 3

if __name__ == '__main__':

	# divide candidate words from preprocessing into quadrants based on a model
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-m', '--model', type=str, default='gpt2-xl') # model to select our quadrants based on
	p = parser.parse_args()

	# set the directories we need
	models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions')
	preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed', p.task)

	# load our preprocessed file --> get the indices of the prediction words
	df_preproc = pd.read_csv(os.path.join(preproc_dir, f'{p.task}_transcript-preprocessed.csv'))
	nwp_idxs = np.where(df_preproc['NWP_Candidate'])[0]

	# select based on model quadrants --> trim down to only the words of interest
	model_results = load_model_results(models_dir, model_name=p.model, task=p.task, window_size=WINDOW_SIZE, top_n=TOP_N)
	model_results.loc[:, 'binary_accuracy'] = model_results['binary_accuracy'].astype(bool)
	model_results = model_results.iloc[nwp_idxs]

	# now divide the words based on quadrants and select words
	df_divide = divide_nwp_dataframe(model_results, accuracy_type=ACCURACY_TYPE, percentile=ACCURACY_PERCENTILE)
	df_selected = select_prediction_words(df_divide, remove_perc=REMOVE_PERC, select_perc=SELECT_PERC, min_spacing_thresh=MIN_SPACING_THRESH)

	# now update df_preproc with our selected indices --> write out
	selected_idxs = df_selected.index
	df_final = df_preproc.copy()
	df_final.loc[selected_idxs, ['entropy_group', 'accuracy_group']] = df_selected[['entropy_group', 'accuracy_group']]

	# set only indicies that were selected to candidates and check
	df_final.loc[~df_final.index.isin(selected_idxs), 'NWP_Candidate'] = False
	assert (df_final.loc[selected_idxs, 'NWP_Candidate'].all())

	df_divide_fn = os.path.join(preproc_dir, f'{p.task}_transcript-model-divided.csv')
	df_divide.to_csv(df_divide_fn, index=False)

	df_final_fn = os.path.join(preproc_dir, f'{p.task}_transcript-selected.csv')
	df_final.to_csv(df_final_fn, index=False)