import os, sys
import argparse
import torch
import pyrootutils

import hydra
from hydra import initialize, compose
from lightning import LightningDataModule, LightningModule

import pandas as pd

sys.path.append('../utils/')
sys.path.append('../modeling/joint-clm-prosody/')

from config import *
from src import utils

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

    results_metrics = ['loss', 'clm_loss', 'prosody_loss', 'perplexity', 'accuracy']
    df_results = []
    
    for i, batch in enumerate(dataloader):
        print (f'Batch {i+1}/{str(len(datamodule.test_dataloader()))}', flush=True)

        with torch.no_grad():
            _, outputs = model.step(batch=batch)
        
        outputs = model._calculate_metrics(outputs, batch, stage=None)

        # cast all to numpy
        metrics = {}

        for metric_name, value in outputs.items():
            if metric_name in ['mae', 'pearson', 'r2']:
                # NEED TO SOLVE MORE THAN 2 VALUES PROBLEM --> SHOULD WE AGGREGATE OVER ALL SAMPLES?

                # metric = getattr(model, f'test_{metric_name}')
                # metric(*value)
                # value = metric.compute()
                # metric.reset()

                continue


            # cast to numpy
            metrics[metric_name] = value.numpy()

        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df_results.append(df_metrics)
        # accuracy = model.calculate_accuracy(logits, labels, mask)
        
        # results.append(accuracy.numpy())

    df_results = pd.concat(df_results).reset_index(drop=True)
    return df_results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-model_name', '--model_name', type=str)
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str)
    parser.add_argument('-overrides', '--overrides', type=str, nargs='+')
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    modeling_dir = os.path.join(BASE_DIR, 'code/modeling/joint-clm-prosody/')
    results_dir = os.path.join(BASE_DIR, 'derivatives/joint-prosody-clm/')

    pyrootutils.setup_root(modeling_dir, indicator=".project-root", pythonpath=True)

    print (f'{p.model_name}', flush=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    ####################################
    ### Initialize hydra config file ###
    ####################################

    # Get relative path --> path for initialize needs to be relative
    config_path = os.path.join(os.path.relpath(modeling_dir, os.getcwd()), 'configs')

    # We set the batch size to 1 because we want an accuracy for each sample
    cfg, model = load_model(config_path, p.ckpt_path, p.overrides)
    cfg.data['batch_size'] = 1

    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, model_name=cfg.model.model_name
    )

    # Get test set for the prosodic prominence dataset
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    #########################################
    ### Load model and calculate accuracy ###
    #########################################

    df_results = get_model_results(model, dataloader)
    df_results['model_name'] = p.model_name
    df_results.to_csv(os.path.join(results_dir, f'{p.model_name}_test-prominence.csv'), index=False)
    