import os, sys
import pandas as pd
import argparse

sys.path.append('../utils/')

from config import *
import analysis_utils as analysis
import prosody_utils as prosody
from text_utils import strip_punctuation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-n_words', '--n_words', type=int, default=5)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    stim_dir = os.path.join(BASE_DIR, 'stimuli/')

    ###############################################
    ######## Load prosody data and process ########
    ###############################################

    # Define column names for prosody data
    prosody_columns = ['stim', 'start', 'end', 'word', 'prominence', 'boundary']

    # Process prosody -- calculate the average prosody over the past n words
    df_prosody = pd.read_csv(os.path.join(stim_dir, 'prosody', f'{p.task}.prom'), sep='\t', names=prosody_columns)
    df_prosody = prosody.calculate_prosody_metrics(df_prosody, n_prev=p.n_words, remove_characters=prosody.REMOVE_WORDS)

    #################################################
    ######## Load transcript and add prosody ########
    #################################################

    # Load transcript -- add prosody information to the transcript
    transcript_fn = os.path.join(stim_dir, 'preprocessed', p.task, f'{p.task}_transcript-selected.csv')
    df_transcript = pd.read_csv(transcript_fn)
    df_transcript = df_transcript.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # Make sure words match between the transcript and prosody dataframes
    words_transcript = df_transcript['word'].str.lower().apply(strip_punctuation)
    words_prosody =  df_prosody['word'].str.lower().apply(strip_punctuation)

    assert all(words_transcript == words_prosody)

    # All words match so merge the dataframes together
    prosody_columns = ['prominence', 'boundary', 'prosody_mean', 'prosody_std', 'relative_prosody', 'relative_norm', 'boundary_mean', 'boundary_std']
    df_transcript.loc[:, prosody_columns] = df_prosody.loc[:, prosody_columns]

    df_transcript.to_csv(transcript_fn.replace('.csv', '_prosody.csv'), index=False)