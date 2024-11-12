import warnings
warnings.filterwarnings("ignore")
    
import os, sys
import argparse
import torch
import pyrootutils

import hydra
from hydra import initialize, compose
from lightning import LightningDataModule, LightningModule

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import pandas as pd
import numpy as np

sys.path.append('../utils/')
sys.path.append('../modeling/joint-clm-prosody/')

from config import *
from dataset_utils import attempt_makedirs
import prosody_analysis_utils as analysis

from src import utils
from src.utils.text_processing import python_remove_punctuation
from src.data.components.collators import collate_fn, encode_and_pad_batch
from src.data.components.datasets import tokenize_text_with_labels, TokenTaggingDataset

def load_model(config_path, ckpt_path, overrides):

    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="train.yaml", overrides=overrides)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Load the model from a checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.model_name, add_prefix_space=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return cfg, tokenizer, model

def get_prosody_model_predictions(batch, model, out_fn=None):
        
    with torch.no_grad():
        _, outputs = model.step(batch=batch)
        logits = outputs['logits'][:, -1, :]
    
    # get the probability of the logits
    probs = F.softmax(logits, dim=-1)

    # if we provide we save logits out
    if out_fn:
        torch.save(logits, out_fn)
    
    return probs

def collate(batch):
    return collate_fn(batch, tokenizer.pad_token_id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', '--model_name', type=str)
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str)
    parser.add_argument('-overrides', '--overrides', type=str, nargs='+')

    # clm extraction parameters 
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-w', '--window_size', type=int, default=25)
    parser.add_argument('-n', '--top_n', type=int, nargs='+', default=5)
    p = parser.parse_args()

    print (f'Task: {p.task}')
    print (f'Window Size: {p.window_size}')
    
    out_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', p.task, 'prosody-models', p.model_name, f'window-size-{p.window_size}')
    logits_dir = os.path.join(SCRATCH_DIR, 'derivatives/model-predictions', p.task, 'prosody-models', p.model_name, f'window-size-{p.window_size}', 'logits')

    attempt_makedirs(out_dir)
    attempt_makedirs(logits_dir)

    ####################################
    ### Initialize hydra config file ###
    ####################################

    print (f'{p.model_name}', flush=True)

    modeling_dir = os.path.join(BASE_DIR, 'code/modeling/joint-clm-prosody/')
    pyrootutils.setup_root(modeling_dir, indicator=".project-root", pythonpath=True)

    
    # Get relative path --> path for initialize needs to be relative
    config_path = os.path.join(os.path.relpath(modeling_dir, os.getcwd()), 'configs')
    cfg, tokenizer, model = load_model(config_path, p.ckpt_path, p.overrides)

    print (f'Model loaded', flush=True)

    ###########################################
    #### Load prosody and transcript info #####
    ###########################################

    # Define column names for prosody data --> remove non-words
    prosody_columns = ['stim', 'start', 'end', 'word', 'prominence', 'boundary']

    df_prosody = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/prosody/', f'{p.task}.prom'), sep='\t', names=prosody_columns)
    df_prosody = df_prosody[~df_prosody['word'].isin(analysis.REMOVE_WORDS)].reset_index(drop=True) # remove non-words

    df_preproc = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/preprocessed/', p.task, f'{p.task}_transcript-preprocessed.csv'))
    df_preproc = df_preproc.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # make sure the words match
    words_preproc = df_preproc['word'].str.lower().apply(python_remove_punctuation)
    words_prosody =  df_prosody['word'].str.lower().apply(python_remove_punctuation)

    assert all(words_preproc == words_prosody)

    # if it matches we can add in prosody as a column
    df_preproc['prominence'] = df_prosody['prominence']
    df_preproc.loc[df_preproc['prominence'] < 0, 'prominence'] = 0

    ###########################################
    #### Create a dataset for processing  #####
    ###########################################

    # create a list of indices that we will iterate through to sample the transcript
    segments = analysis.get_segment_indices(n_words=len(df_preproc), window_size=p.window_size)[:-1]
    inputs = [analysis.transcript_to_input(df_preproc, segment, add_punctuation=True) for segment in segments]
    inputs, labels = zip(*inputs)

    # now create the dataset out of the labels --> buffer the missing samples to keep it aligned with the transcript
    dataset = TokenTaggingDataset(inputs, labels, tokenizer, model_name=cfg.model.model_name, remove_punctuation=False, buffer_missing_samples=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate)

    ###########################################
    #### Load word models and prep stats ######
    ###########################################

    # load a word-level model --> we use glove here
    # first function downloads the model if needed, second function loads it as gensim format
    word_models = {model_name: analysis.load_word_model(model_name=model_name, cache_dir=CACHE_DIR) for model_name in analysis.WORD_MODELS.keys()}

    # add the first word to the dataframe --> we don't run NWP on this as there is no context
    # to condition, nor do we have humans do it
    df = analysis.create_results_dataframe()
    first_word = df_preproc.iloc[0]['word'].lower()
    df.loc[len(df)] = {'ground_truth_word': first_word}

    # set up variables to be used in the loop
    df_stack = {str(n): [df] for n in p.top_n}
    prev_probs = None

    #################################################
    #### Go through the dataloader and process ######
    #################################################

    for i, batch in enumerate(dataloader):

        ground_truth_index = segments[i][-1] + 1
        ground_truth_word = df_preproc.loc[ground_truth_index, 'word']

        # run the inputs through the model, get predictive distribution, and save out the logits
        # if the next word is a prediction word save logits
        if df_preproc.loc[ground_truth_index, 'NWP_Candidate']: # and p.model_name == 'gpt2-xl':
            logits_fn = os.path.join(logits_dir, f'{p.task}_window-size-{p.window_size}_logits-{str(ground_truth_index).zfill(5)}.pt')
        else:
            logits_fn = None

        # we've buffered the samples and we're gonna wait for the real one
        if any(batch['input_text']):
            probs = get_prosody_model_predictions(batch, model, out_fn=logits_fn)

            # now given the outputs of the model, run our stats of interest
            for n in p.top_n:
                segment_stats = analysis.get_model_statistics(ground_truth_word, probs, tokenizer, prev_probs=prev_probs, word_models=word_models, top_n=n)
                df_stack[str(n)].append(segment_stats)

            # now that we've run our stats, set the previous distribution to the one we just ran
            prev_probs = probs
        
        else:
            # append blank frames for the current rows
            for n in p.top_n:
                df = analysis.create_results_dataframe()
                df.loc[len(df)] = {'ground_truth_word': ground_truth_word}
                df_stack[str(n)].append(df)
        
        print (f'Processed segment {i+1}/{len(segments)}', flush=True)

    for n in p.top_n:
        df_results = pd.concat(df_stack[str(n)])
        df_results.to_csv(os.path.join(out_dir, f'task-{p.task}_window-size-{p.window_size}_top-{n}.csv'), index=False)
