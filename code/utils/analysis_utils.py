import os, sys
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

######################################
########## Prosody metrics ###########
######################################

REMOVE_WORDS = ["sp", "br", "lg", "cg", "ls", "ns", "sl", "ig", "{sp}", "{br}", "{lg}", 
 "{cg}", "{ls}", "{ns}", "{sl}", "{ig}", "SP", "BR", "LG", "CG", "LS",
 "NS", "SL", "IG", "{SP}", "{BR}", "{LG}", "{CG}", "{LS}", "{NS}", "{SL}", "{IG}", "pause"]

def calculate_prosody_metrics(df_prosody, n_prev=3, remove_characters=[], zscore=False):
    # Extract raw values
    prosody_raw = df_prosody['prominence'].to_numpy()
    boundary_raw = df_prosody['boundary'].to_numpy()

    if zscore:
        prosody_raw = stats.zscore(prosody_raw)
    
    # get mean of past n_words
    indices = np.arange(len(prosody_raw))
    start_idxs = indices - n_prev
    start_idxs[start_idxs < 0] = 0

    # go through the past x words 
    all_items = []
    
    for idx in start_idxs:
        # get the prosody of the n_prev words
        if idx >= n_prev:
            n_prev_prosody =  prosody_raw[idx:idx+n_prev]
            n_prev_boundary =  boundary_raw[idx:idx+n_prev]
    
            # get mean and std of n_prev words prosody
            prosody_mean = n_prev_prosody.mean()
            prosody_std = n_prev_prosody.std()
    
            # get linear fit to n_prev words
            slope, _ = np.polyfit(np.arange(n_prev), n_prev_prosody, 1)

            relative = prosody_raw[idx+n_prev] - prosody_mean
            relative_norm = relative / prosody_std

            # get mean and std of n_prev prosodic boundaries
            boundary_mean = n_prev_boundary.mean()
            boundary_std = n_prev_boundary.std()
            
        else:
            prosody_mean = prosody_std = slope = relative = relative_norm = np.nan
            boundary_mean = boundary_std = np.nan
        
        all_items.append(
            (prosody_mean, prosody_std, slope, relative, relative_norm, boundary_mean, boundary_std)
        )

    prosody_mean, prosody_std, slope, relative_prosody, relative_norm, boundary_mean, boundary_std = zip(*all_items)

    df_prosody['prosody_mean'] = prosody_mean
    df_prosody['prosody_std'] = prosody_std
    df_prosody['prosody_slope'] = slope
    df_prosody['relative_prosody'] = relative_prosody
    df_prosody['relative_norm'] = relative_norm
    df_prosody['boundary_mean'] = boundary_mean
    df_prosody['boundary_std'] = boundary_std

    # remove non-words
    df_prosody = df_prosody[~df_prosody['word'].isin(remove_characters)].reset_index(drop=True)
    
    return df_prosody