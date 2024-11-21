import os, sys
import argparse
import pandas as pd

sys.path.append('../utils/')

from config import *
import dataset_utils as utils
import analysis_utils as analysis

EXPERIMENT_NAME = 'next-word-prediction'

N_ORDERS = {
    'black': 4,
    'wheretheressmoke': 3,
    'howtodraw': 3
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-v', '--experiment_version', type=str)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    # Sourced for aggregating data across subjects
    results_dir = os.path.join(BASE_DIR, 'experiments',  EXPERIMENT_NAME, 'cleaned-results', p.experiment_version)
    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    audio_dir = os.path.join(BASE_DIR, 'stimuli/cut_audio/', p.experiment_version)
    behavioral_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/') # where we will write our data

    utils.attempt_makedirs(behavioral_dir)

    ###############################################
    ####### Load transcript with prosody info #####
    ###############################################

    # Load transcript --> use the version that we include prosody values within
    df_transcript = pd.read_csv(os.path.join(preproc_dir, p.task, f'{p.task}_transcript-selected_prosody.csv'))
    df_transcript = df_transcript.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    prosody_columns = [
        'prominence', 'prominence_mean', 'prominence_std', 
        'relative_prominence', 'relative_prominence_norm',
        'boundary', 'boundary_mean', 'boundary_std', 
    ]

    ########################################################
    #### Aggregate results across audio/text modalities ####
    ########################################################

    df_aggregated_results = []

    for modality in ['audio', 'text']:

        # Process data for modality
        df_modality = analysis.aggregate_participant_responses(results_dir, audio_dir, task=p.task, modality=modality, n_orders=N_ORDERS[p.task])
        df_modality = analysis.calculate_response_accuracy(df_modality)

        # Add in prosody columns and add to the list
        df_modality.loc[:, prosody_columns] = df_transcript.loc[df_modality['word_index'], prosody_columns].reset_index(drop=True)
        df_aggregated_results.append(df_modality)

    # Concatenate and save compiled cleaned
    df_aggregated_results = pd.concat(df_aggregated_results).reset_index(drop=True)

    out_fn = os.path.join(behavioral_dir, f'task-{p.task}_group-cleaned-behavior.csv')
    df_aggregated_results.to_csv(out_fn, index=False)

    ########################################################
    ###### Lemmatize results and recalculate accuracy ######
    ########################################################

    # Lemmatize responses & ground truth --> save results 
    df_lemmatized_results = analysis.lemmatize_responses(df_aggregated_results, df_transcript, response_column='response')
    df_lemmatized_results = analysis.lemmatize_responses(df_lemmatized_results, df_transcript, response_column='ground_truth')

    # Recalculate response accuracy for lemmatized data
    df_lemmatized_results = analysis.calculate_response_accuracy(df_lemmatized_results)
    
    out_fn = os.path.join(behavioral_dir, f'task-{p.task}_group-cleaned-behavior_lemmatized.csv')
    df_lemmatized_results.to_csv(out_fn, index=False)

