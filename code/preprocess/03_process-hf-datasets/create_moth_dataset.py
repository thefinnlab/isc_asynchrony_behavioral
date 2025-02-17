import os, sys, glob
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from praatio import textgrid as tgio

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils
import prosody_utils as prosody
from text_utils import strip_punctuation
from preproc_utils import dataframe_to_textgrid, cut_audio_segments

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-w', '--window_size', type=int, default=25)
    p = parser.parse_args()

    # Set directories
    stim_dir = os.path.join(BASE_DIR, 'stimuli')
    out_dir = os.path.join(DATASETS_DIR, 'nlp-datasets/pfka-moth-stories/')

    audio_out_dir = os.path.join(out_dir, 'audio', p.task)
    textgrid_out_dir = os.path.join(out_dir, 'textgrids', p.task)
    prosody_out_dir = os.path.join(out_dir, 'prosody', p.task)

    utils.attempt_makedirs(audio_out_dir)
    utils.attempt_makedirs(textgrid_out_dir)

    # Grab the preprocessed data in CSV form (has casing)
    df_preproc = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/preprocessed/', p.task, f'{p.task}_transcript-preprocessed.csv'))
    df_transcript = df_preproc.copy().rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # Grab the prosody .prom file data
    prosody_columns = ['stim', 'start', 'end', 'word', 'prominence', 'boundary']
    df_prosody = pd.read_csv(os.path.join(stim_dir, 'prosody', f'{p.task}.prom'), sep='\t', names=prosody_columns)
    df_prosody = df_prosody[~df_prosody['word'].isin(prosody.REMOVE_WORDS)].reset_index(drop=True) # emove non-words

    # Make sure words match between the transcript and prosody dataframes
    words_transcript = df_transcript['word'].str.lower().apply(strip_punctuation)
    words_prosody =  df_prosody['word'].str.lower().apply(strip_punctuation)

    assert all(words_transcript == words_prosody)

    df_prosody['word'] = df_transcript['word']

    # Get all segments except the first (which can't be cut) and the last (which isn't a word in the transcript)
    segments = prosody.get_segment_indices(n_words=len(df_preproc), window_size=p.window_size)[:-1]
    segment_idxs = [[min(segment), max(segment)]for segment in segments]

    # Cut all audio segments
    base_audio_fn = glob.glob(os.path.join(stim_dir, 'audio', f'*{p.task}*.wav'))[0]
    audio_fns, df_segments = cut_audio_segments(df_preproc, p.task, base_audio_fn, audio_out_dir, segment_idxs, target_sr=16000)

    print (f'Starting processing for {p.task}', flush=True)

    for segment, audio_fn in tqdm(zip(segments, audio_fns), desc=f"Making TextGrid files"):

        # Grab the current segment
        df = df_preproc.iloc[segment]

        # Normalize time to the start of the clip (e.g., make the first onset here 0s)
        df.loc[:, ['Onset', 'Offset']] = df.loc[:, ['Onset', 'Offset']] - df.iloc[0]['Onset']

        # Make a textgrid file
        textgrid_fn = os.path.basename(audio_fn).replace('.wav', '.TextGrid')
        textgrid_fn = os.path.join(textgrid_out_dir, textgrid_fn)

        # Use the written audio file to make the text grid times and save
        tg = dataframe_to_textgrid(df, audio_fn, tier_name='words')
        tg.save(textgrid_fn, 'long_textgrid', True)

        # Make prosody files 
        segment_prosody = df_prosody.iloc[segment] 

        prosody_fn = os.path.basename(audio_fn).replace('.wav', '.prom')
        prosody_fn = os.path.join(prosody_out_dir, prosody_fn)
        segment_prosody.to_csv(prosody_fn, header=False, sep='\t', index=False)
    # sys.exit(0)