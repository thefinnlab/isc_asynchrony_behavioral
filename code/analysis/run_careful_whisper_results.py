import warnings
warnings.filterwarnings("ignore")
    
import os, sys
import argparse
import torch
import pyrootutils

import hydra
from hydra import initialize, compose
from lightning import LightningDataModule, LightningModule

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import pandas as pd
import numpy as np

sys.path.append('../utils/')
sys.path.append('../modeling/joint-clm-prosody/')

from config import *
from dataset_utils import attempt_makedirs
import prosody_utils as prosody

from src import utils
from src.data.components.audio_text_dataset import AudioTextDataset
from src.data.components.collators import audio_text_collator

def load_model(config_path, ckpt_path, overrides):
    
    with initialize(version_base="1.3", config_path=config_path):
      cfg = compose(config_name="train.yaml", overrides=overrides)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Load the model from a checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return cfg, model

def get_model_results(model, dataloader):

    results_metrics = ['loss', 'perplexity', 'accuracy']
    df_results = []
    
    for i, batch in enumerate(dataloader):
        print (f'Batch {i+1}/{len(dataloader)}', flush=True)

        with torch.no_grad():
            outputs = model.step(batch=batch)
        
        outputs = model._calculate_metrics(outputs, batch, stage=None)

        # cast all to numpy
        metrics = {}

        for metric_name, value in outputs.items():

            # cast to numpy
            metrics[metric_name] = value.numpy()

        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df_results.append(df_metrics)

    df_results = pd.concat(df_results).reset_index(drop=True)
    return df_results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-s', '--split', type=str)
    parser.add_argument('-model_name', '--model_name', type=str)
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str)
    parser.add_argument('-overrides', '--overrides', type=str, nargs='+')
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    modeling_dir = os.path.join(BASE_DIR, 'code/modeling/joint-clm-prosody/')
    results_dir = os.path.join(BASE_DIR, f'derivatives/results/careful-whisper/{p.dataset}/')

    if p.dataset == 'gigaspeech-m':
        results_dir = os.path.join(BASE_DIR, f'derivatives/results/careful-whisper/{p.dataset}/')
        dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset.replace('-', '/'))
        cache_dir = os.path.join(SCRATCH_DIR, 'nlp-datasets', p.dataset.replace('-', '/'))
    else:
        dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)
        cache_dir = os.path.join(SCRATCH_DIR, 'nlp-datasets', p.dataset)

    pyrootutils.setup_root(modeling_dir, indicator=".project-root", pythonpath=True)

    print (f'{p.model_name}', flush=True)

    attempt_makedirs(results_dir)
    
    ####################################
    ### Initialize hydra config file ###
    ####################################

    # Get relative path --> path for initialize needs to be relative
    config_path = os.path.join(os.path.relpath(modeling_dir, os.getcwd()), 'configs')

    # We set the batch size to 1 because we want an accuracy for each sample
    cfg, model = load_model(config_path, p.ckpt_path, p.overrides)
    # cfg.data['batch_size'] = 1

    print (dataset_dir, flush=True)
    print (p.split, flush=True)

    ####################################
    ###### Initialize dataloader #######
    ####################################

    # create dataset for the test split
    dataset = AudioTextDataset(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        split=p.split,
    )

    dataset.preprocess_data()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=audio_text_collator,
        shuffle=False,
    )

    #########################################
    ### Load model and calculate accuracy ###
    #########################################

    if p.dataset != 'pfka-moth-stories':
        df_results = get_model_results(model, dataloader)
        df_results['model_name'] = p.model_name
        df_results.to_csv(os.path.join(results_dir, f'{p.model_name}_test.csv'), index=False)
    else:

        dataset._initialize_models()

        TOP_N = [1, 5]
        WINDOW_SIZE = 25

        out_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', p.split, 'careful-whisper', p.model_name, f'window-size-{str(WINDOW_SIZE).zfill(5)}')
        logits_dir = os.path.join(SCRATCH_DIR, 'derivatives/model-predictions', p.split, 'careful-whisper', p.model_name, f'window-size-{str(WINDOW_SIZE).zfill(5)}', 'logits')

        attempt_makedirs(out_dir)
        attempt_makedirs(logits_dir)

        # Load our preprocessed dataframe
        df_preproc = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/preprocessed/', p.split, f'{p.split}_transcript-preprocessed.csv'))
        df_preproc = df_preproc.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

        ###########################################
        #### Load word models and prep stats ######
        ###########################################

        print ('Loading word models...', flush=True)

        # load a word-level model --> we use glove here
        # first function downloads the model if needed, second function loads it as gensim format
        word_models = {model_name: prosody.load_word_model(model_name=model_name, cache_dir=CACHE_DIR) for model_name in prosody.WORD_MODELS.keys()}

        # add the first word to the dataframe --> we don't run NWP on this as there is no context
        # to condition, nor do we have humans do it
        df = prosody.create_results_dataframe()
        first_word = df_preproc.iloc[0]['word'].lower()
        df.loc[len(df)] = {'ground_truth_word': first_word}

        # set up variables to be used in the loop
        df_stack = {str(n): [df] for n in TOP_N}
        prev_probs = None

        #################################################
        #### Go through the dataloader and process ######
        #################################################

        for i, batch in enumerate(dataloader):

            ground_truth_index = i + 1
            ground_truth_word = df_preproc.loc[ground_truth_index, 'word']

            with torch.no_grad():
                outputs = model.step(batch=batch)

            logits = outputs['logits'][:, -1, :]

            # run the inputs through the model, get predictive distribution, and save out the logits
            # if the next word is a prediction word save logits
            if df_preproc.loc[ground_truth_index, 'NWP_Candidate']: # and p.model_name == 'gpt2-xl':
                logits_fn = os.path.join(logits_dir, f'{p.split}_window-size-{str(WINDOW_SIZE).zfill(5)}_logits-{str(ground_truth_index).zfill(5)}.pt')
                torch.save(logits, logits_fn)
            else:
                logits_fn = None

            # convert logits to probability
            probs = F.softmax(logits, dim=-1)

            # now given the outputs of the model, run our stats of interest
            for n in TOP_N:
                segment_stats = prosody.get_model_statistics(ground_truth_word, probs, dataset.text_tokenizer, prev_probs=prev_probs, word_models=word_models, top_n=n)
                df_stack[str(n)].append(segment_stats)

            # now that we've run our stats, set the previous distribution to the one we just ran
            prev_probs = probs

            print (f'Processed segment {i+1}/{len(dataloader)}', flush=True)

        for n in TOP_N:
            df_results = pd.concat(df_stack[str(n)]).reset_index(drop=True)
            df_results.to_csv(os.path.join(out_dir, f'task-{p.split}_window-size-{str(WINDOW_SIZE).zfill(5)}_top-{n}.csv'), index=False)