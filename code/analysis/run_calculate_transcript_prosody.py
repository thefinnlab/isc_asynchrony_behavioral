import os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../utils/')

from config import *
from text_utils import strip_punctuation

######################################
########## Prosody metrics ###########
######################################

REMOVE_WORDS = ["sp", "br", "lg", "cg", "ls", "ns", "sl", "ig", "{sp}", "{br}", "{lg}", 
 "{cg}", "{ls}", "{ns}", "{sl}", "{ig}", "SP", "BR", "LG", "CG", "LS",
 "NS", "SL", "IG", "{SP}", "{BR}", "{LG}", "{CG}", "{LS}", "{NS}", "{SL}", "{IG}", "pause"]

def calculate_prosody_metrics(df_prosody, n_prev=3, remove_characters=[], zscore=False):

    df = df_prosody.copy()

    # Extract raw values
    prosody_raw = df['prominence'].to_numpy()
    boundary_raw = df['boundary'].to_numpy()

    if zscore:
        prosody_raw = stats.zscore(prosody_raw)
    
    # get mean of past n_words
    indices = np.arange(len(prosody_raw))
    start_idxs = indices - n_prev

    # go through the past x words 
    all_items = []
    
    for idx in tqdm(start_idxs):
        # get the prosody of the n_prev words
        if idx >= 0:
            n_prev_prosody = prosody_raw[idx:idx+n_prev]
            n_prev_boundary = boundary_raw[idx:idx+n_prev]
            
            # get mean and std of n_prev words prosody
            prosody_mean = n_prev_prosody.mean()
            prosody_std = n_prev_prosody.std()

            relative_prosody = prosody_raw[idx+n_prev] - prosody_mean
            relative_prosody_norm = relative_prosody / prosody_std

            # get mean and std of n_prev prosodic boundaries
            boundary_mean = n_prev_boundary.mean()
            boundary_std = n_prev_boundary.std()
            
        else:
            prosody_mean = prosody_std = relative_prosody = relative_prosody_norm = np.nan
            boundary_mean = boundary_std = np.nan
        
        all_items.append(
            (prosody_mean, prosody_std, relative_prosody, relative_prosody_norm, boundary_mean, boundary_std)
        )

    prosody_mean, prosody_std, relative_prosody, relative_prosody_norm, boundary_mean, boundary_std = zip(*all_items)

    df['prominence_mean'] = prosody_mean
    df['boundary_mean'] = boundary_mean
    # df_prosody['prominence_std'] = prosody_std
    # df_prosody['relative_prominence'] = relative_prosody
    # df_prosody['relative_prominence_norm'] = relative_prosody_norm

    # df_prosody['boundary_std'] = boundary_std

    # remove non-words
    # df = df[~df['word'].isin(remove_characters)].reset_index(drop=True)
    
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-num_words', '--num_words', type=int, nargs='+', default=5)
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

    extract_columns = ['prominence_mean', 'boundary_mean']
    all_renamed = []

    for n_words in p.num_words:
        # create the columns to rename to
        renamed_columns = {x: f'{x}_words{n_words}' for x in extract_columns}

        df_prosody_words = calculate_prosody_metrics(df_prosody, n_prev=n_words, remove_characters=REMOVE_WORDS)

        # Extract the columns we care about --> rename those columns
        df_prosody_words = df_prosody_words[extract_columns]
        df_prosody_words = df_prosody_words.rename(columns=renamed_columns)

        # Grab the renamed columns --> extract them and join them into the dataframe
        renamed_columns = list(renamed_columns.values())
        df_prosody_words = df_prosody_words[renamed_columns]
        df_prosody = df_prosody.join(df_prosody_words)

        # Store them for later
        all_renamed += renamed_columns

    # Remove non-words
    df_prosody = df_prosody[~df_prosody['word'].isin(REMOVE_WORDS)].reset_index(drop=True)

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
    prosody_columns = [
        'prominence', #'prominence_mean', 
        # 'prominence_std', 'relative_prominence', 'relative_prominence_norm',
        'boundary', #'boundary_mean', #'boundary_std', 
    ]

    prosody_columns += all_renamed

    df_transcript.loc[:, prosody_columns] = df_prosody.loc[:, prosody_columns]

    df_transcript.to_csv(transcript_fn.replace('.csv', '_prosody.csv'), index=False)