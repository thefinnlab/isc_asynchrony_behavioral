import os, sys
import argparse
import numpy as np
import pandas as pd

sys.path.append('../utils/')

from config import *
import dataset_utils as utils
import analysis_utils as analysis
from tommy_utils.nlp import load_word_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-m', '--model_names', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model_name', type=str, default='fasttext')
    parser.add_argument('-window_size', '--window_size', type=int, default=25)
    parser.add_argument('-prosody', '--prosody', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    # Sourced for aggregating data across subjects
    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions')
    logits_dir = models_dir.replace(BASE_DIR, SCRATCH_DIR)

    ###############################################
    ####### Load transcript with prosody info #####
    ###############################################

    # Load transcript --> use the version that we include prosody values within
    df_transcript = pd.read_csv(os.path.join(preproc_dir, p.task, f'{p.task}_transcript-selected_prosody.csv'))
    df_transcript = df_transcript.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # Need to load the preprocessed transcript that has all words to get the model quadrants
    df_preproc_transcript = pd.read_csv(os.path.join(preproc_dir, p.task, f'{p.task}_transcript-preprocessed.csv'))
    candidate_rows = np.where(df_preproc_transcript['NWP_Candidate'])[0]

    prosody_columns = [
        'prominence', 'prominence_mean', 'prominence_std', 
        'relative_prominence', 'relative_prominence_norm',
        'boundary', 'boundary_mean', 'boundary_std', 
    ]

    # select the columns that we want to save out for gross-comparison
    combined_columns = [
        'modality', 'word_index', 'top_pred', 'ground_truth', 'accuracy', 
        f'{p.word_model_name}_top_word_accuracy', 'top_prob', 'predictability', 
        'entropy', 'entropy_group', 'accuracy_group'
    ]

    combined_columns = combined_columns + prosody_columns

    ###########################################################
    ####### Load word model used for continuous accuracy ######
    ###########################################################

    word_model = load_word_model(model_name=p.word_model_name, cache_dir=CACHE_DIR)
    word_model_info = (p.word_model_name, word_model)

    # ########################################################
    # ############ Load human results and analyze ############
    # ########################################################

    # # Analyze results from cleaned data 
    # df_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior.csv'))
    # df_analyzed_results = analysis.analyze_human_results(df_transcript, df_results, word_model_info, window_size=p.window_size, top_n=None, drop_rt=None)

    # # Add in prosody columns and add to the list
    # out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_human.csv')
    # df_analyzed_results.loc[:, prosody_columns] = df_transcript.loc[df_analyzed_results['word_index'], prosody_columns].reset_index(drop=True)
    # df_analyzed_results.to_csv(out_fn, index=False)
    
    # ### Now do model data
    # df_all_models = []

    # for model_name in p.model_names:
    #     df_model = analysis.analyze_model_accuracy(df_transcript, word_model_info=word_model_info, models_dir=models_dir, model_name=model_name, task=p.task, candidate_rows=candidate_rows, lemmatize=False)

    #     # Add in prosody columns and add to the list
    #     df_model.loc[:, prosody_columns] = df_transcript.loc[df_model['word_index'], prosody_columns].reset_index(drop=True)
    #     df_all_models.append(df_model)

    # # Concatenate models and then add to the humans dataframe
    # df_all_models = pd.concat(df_all_models).reset_index(drop=True)

    # df_combined = pd.concat([
    #     df_analyzed_results.loc[:, combined_columns], 
    #     df_all_models.loc[:, combined_columns]
    # ]).reset_index(drop=True)

    # out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{p.window_size}_human-model.csv')
    # df_combined.to_csv(out_fn, index=False)

    # ########################################################
    # ########## Repeat process for lemmatized words #########
    # ########################################################

    # # Analyze human lemmatized data
    # df_lemmatized_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior_lemmatized.csv'))
    # df_analyzed_lemmatized = analysis.analyze_human_results(df_transcript, df_lemmatized_results, word_model_info, window_size=p.window_size, top_n=None, drop_rt=None)

    # # Add in prosody columns and add to the list
    # out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_human-lemmatized.csv')
    # df_analyzed_lemmatized.loc[:, prosody_columns] = df_transcript.loc[df_analyzed_lemmatized['word_index'], prosody_columns].reset_index(drop=True)
    # df_analyzed_lemmatized.to_csv(out_fn, index=False)

    # ### Now do model data
    # df_lemmatized_models = []

    # for model_name in p.model_names:
    #     df_model = analysis.analyze_model_accuracy(df_transcript, word_model_info=word_model_info, models_dir=models_dir, model_name=model_name, task=p.task, candidate_rows=candidate_rows, lemmatize=True)

    #     # Add in prosody columns and add to the list
    #     df_model.loc[:, prosody_columns] = df_transcript.loc[df_model['word_index'], prosody_columns].reset_index(drop=True)
    #     df_lemmatized_models.append(df_model)

    # # Concatenate models and then add to the humans dataframe
    # df_lemmatized_models = pd.concat(df_lemmatized_models).reset_index(drop=True)

    # out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{p.window_size}_human-model-lemmatized.csv')

    # df_lemmatized_combined = pd.concat([
    #     df_analyzed_lemmatized.loc[:, combined_columns], 
    #     df_lemmatized_models.loc[:, combined_columns]
    # ]).reset_index(drop=True)

    # df_lemmatized_combined.to_csv(out_fn, index=False)

    ########################################################
    ########### Conduct distribution comparison ############
    ########################################################

    df_lemmatized_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior_lemmatized.csv'))

    df_all_comparisons = []

    for model_name in p.model_names:
        print (model_name)

        df_comparison = analysis.compare_human_model_distributions(df_lemmatized_results, word_model_info, models_dir=logits_dir, model_name=model_name, task=p.task, lemmatize=True)
        df_comparison.loc[:, prosody_columns] = df_transcript.loc[df_comparison['word_index'], prosody_columns].reset_index(drop=True)
        df_all_comparisons.append(df_comparison)

    if p.prosody:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{p.window_size}_human-prosody-model-distributions-lemmatized.csv')
    else:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{p.window_size}_human-model-distributions-lemmatized.csv')
    
    df_all_comparisons = pd.concat(df_all_comparisons).reset_index(drop=True)
    df_all_comparisons.to_csv(out_fn, index=False)