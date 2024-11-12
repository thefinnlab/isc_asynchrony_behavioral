import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
from itertools import product
import subprocess

sys.path.append('../utils/')

from config import *
import dataset_utils as utils

PARTITION = 'standard'
TIME = '12:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'

if __name__ == '__main__':

    task_names = ['black', 'wheretheressmoke', 'howtodraw']
    dataset_names = ['helsinki', 'gigaspeech']

    window_sizes = [25, 100]
    top_ns = [1, 5]
    top_ns = " ".join(str(v) for v in top_ns)
    

    MODELS_DIR = os.path.join(BASE_DIR, 'code/modeling/joint-clm-prosody/')

    MODEL_ARGS = {
        'helsinki': {
            # Prosody model + Joint loss: training with both CLM loss & prosody prediction objectives + adding prosodic information
            'helsinki-prosody_scratch-gpt2_joint-loss_prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-09-12/07-41-15/checkpoints/epoch_014.ckpt'),
                'overrides': [f"model.loss_mode=joint", f"model.pretrained=False", f"model.use_prosody_embeddings=True"]
            },

            # Prosody model + CLM loss: training with only CLM loss adding prosodic information
            'helsinki-prosody_scratch-gpt2_clm-loss_prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-09-11/17-06-38/checkpoints/epoch_014.ckpt'),
                'overrides': [f"model.loss_mode=clm", f"model.pretrained=False", f"model.use_prosody_embeddings=True"]
            },

            # Baseline model: training with only CLM loss without prosody embeddings
            'helsinki-prosody_scratch-gpt2_clm-loss_no-prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-10-24/14-26-40/checkpoints/epoch_014.ckpt'), 
                'overrides': [f"model.loss_mode=clm", f"model.pretrained=False", f"model.use_prosody_embeddings=False"]
            }, 
        },

        ################### GIGASPEECH ############

        'gigaspeech' : {

            # Prosody model + Joint loss: training with both CLM loss & prosody prediction objectives + adding prosodic information
            'gigaspeech-prosody_scratch-gpt2_joint-loss_prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-11-01/18-06-45/checkpoints/epoch_014.ckpt'),
                'overrides': [f"model.loss_mode=joint", f"model.pretrained=False", f"model.use_prosody_embeddings=True"]
            },

            # Prosody model + CLM loss: training with only CLM loss adding prosodic information
            'gigaspeech-prosody_scratch-gpt2_clm-loss_prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-11-01/16-04-33/checkpoints/epoch_013.ckpt'),
                'overrides': [f"model.loss_mode=clm", f"model.pretrained=False", f"model.use_prosody_embeddings=True"]
            },

            # Baseline model: training with only CLM loss without prosody embeddings
            'gigaspeech-prosody_scratch-gpt2_clm-loss_no-prosody-embed': {
                'ckpt_path': os.path.join(MODELS_DIR, 'logs/train/runs/2024-11-01/16-04-14/checkpoints/epoch_013.ckpt'), 
                'overrides': [f"model.loss_mode=clm", f"model.pretrained=False", f"model.use_prosody_embeddings=False"]
            }, 
        }
    }

    # grab the tasks
    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'run_prosody_model_predictions.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'

    # Sub in the conda environment
    job_string = job_string.replace('dark_matter', 'prosody')
    job_num = 0

    for dataset, task, window in product(dataset_names, task_names, window_sizes):

        experiment = [f"experiment={dataset}_prosody.yaml"]

        for i, (model_name, cfg_args) in enumerate(MODEL_ARGS[dataset].items()):

            overrides = " ".join(experiment + cfg_args['overrides'])

            cmd = ''.join((
                f"{job_string} -t {task} -w {window} -n {top_ns} "
                f"-model_name {model_name} -ckpt_path {cfg_args['ckpt_path']} "
                f"-overrides {overrides}"
            ))

            if job_num in [14, 32]:
                all_cmds.append(cmd)
            job_num += 1
    
    dsq_base_string = f'dsq_prosody_model_predictions'
    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/behavioral/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/joblists')

    utils.attempt_makedirs(logs_dir)
    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblists_dir)

    joblist_fn = os.path.join(joblists_dir, f'run_prosody_model_predictions.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))
    
    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)
    
    # subprocess.run('module load dSQ', shell=True)
    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)