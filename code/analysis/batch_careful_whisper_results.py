import warnings
warnings.filterwarnings("ignore")

import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
from itertools import product
import subprocess

sys.path.append('../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../modeling/careful-whisper/scripts/')

import utils

PARTITION = 'preemptable'
TIME = '12:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
EXCLUDE = 'q04'
ACCOUNT = 'dbic'

DATASET_INFO = {
    'gigaspeech-m': {
        'splits': ['test'],
        'data_config':[
            'data.dataset_name=gigaspeech-m',
            'data.data_dir=\${paths.data_dir}/gigaspeech/m',
            'data.cache_dir=\${paths.cache_dir}/nlp-datasets/gigaspeech/m',
    ]},

    'libritts-r': {
        'splits': ['test-clean'],
        'data_config': [
            'data.dataset_name=libritts-r',
            'data.data_dir=\${paths.data_dir}/libritts-r',
            'data.cache_dir=\${paths.cache_dir}/nlp-datasets/libritts-r',
    ]},

    'tedlium': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=tedlium',
            'data.data_dir=\${paths.data_dir}/tedlium',
            'data.cache_dir=\${paths.cache_dir}/nlp-datasets/tedlium',
    ]},

    'peoples-speech': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=peoples-speech',
            'data.data_dir=\${paths.data_dir}/peoples-speech',
            'data.cache_dir=\${paths.cache_dir}/nlp-datasets/peoples-speech',
    ]},

    'pfka-moth-stories': {
        'splits': ['black', 'wheretheressmoke', 'howtodraw'],
        'data_config': [
            'data.dataset_name=pfka-moth-stories',
            'data.data_dir=\${paths.data_dir}/pfka-moth-speech',
            'data.cache_dir=\${paths.cache_dir}/nlp-datasets/pfka-moth-speech',
    ]},

    'lrs3': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=lrs3',
            'data.data_dir=\${paths.data_dir}/lrs3',
    ]},

    'voxceleb2': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=voxceleb2',
            'data.data_dir=\${paths.data_dir}/voxceleb2',
    ]},

    'avspeech': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=avspeech',
            'data.data_dir=\${paths.data_dir}/avspeech',
    ]},

    'av-combined': {
        'splits': ['test'],
        'data_config': [
            'data.dataset_name=av-combined',
            'data.data_dir=\${paths.data_dir}/av-combined',
    ]},
}

def create_model_variations(base_configs, subset_percentages=None):
    """Create model config variations for different subset sizes."""
    variations = {}
    
    for model_name, config in base_configs.items():        
        # Add subset versions
        if subset_percentages is not None:
            for subset in subset_percentages:
                subset_name = f"{model_name}_subset-{str(subset).zfill(3)}"
                subset_config = config.copy()
                subset_config.append(f"data.subset_percentage={subset}")
                variations[subset_name] = subset_config
        else:
            # Full dataset version
            variations[model_name] = config.copy()
            
    return variations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-train', '--train_dataset', type=str)
    parser.add_argument('-test', '--test_dataset', type=str, default=None)
    parser.add_argument('-subsets', '--subsets', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    if not p.test_dataset:
        p.test_dataset = p.train_dataset

    EXPERIMENT_NAME = 'careful_whisper'
    MODELS_DIR = os.path.join(BASE_DIR, 'code/modeling/careful-whisper/')
    CKPTS_DIR = os.path.join(MODELS_DIR, f'logs/train/careful-whisper/{p.train_dataset}/')

    MODEL_CONFIGS = {

        # General GPT2-esque model
        'text-careful-whisper_no-xattn': [
            f"model.config.cross_attention=False",
            f"model.config.use_causal_cross_attention=False",
        ],

        # Whisper w/ CLM integration
        'audio-careful-whisper_causal-xattn': [
            f"model.config.cross_attention=True",
            f"model.config.use_causal_cross_attention=True",

            # Add in dropout and position embedding
            f"model.config.context_embed_dropout=0.1",
            f"model.config.context_pos_embed=True",
        ],

        # Whisper w/ CLM integration
        'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp': [
            f"model.config.cross_attention=True",
            f"model.config.use_causal_cross_attention=True",

            # Prosody embedding information
            f"model.config.context_type=audiovisual_features",
            f"model.config.context_embed_dropout=0.1",
            f"model.config.context_pos_embed=True",
        ],

        # Whisper w/ CLM integration
        'prosody-careful-whisper_causal-xattn': [
            f"model.config.cross_attention=True",
            f"model.config.use_causal_cross_attention=True",

            # Prosody embedding information
            f"model.config.context_type=prominence",
            f"model.config.context_dim=1",
            f"model.config.context_embed_dropout=0.1",
            f"model.config.context_pos_embed=True",

        ],
    }

    # Set up dataset info as the test dataset
    dataset_config = ' '.join(DATASET_INFO[p.test_dataset]['data_config'])
    splits = DATASET_INFO[p.test_dataset]['splits']

    # # Log space 2 - 25% 
    subset_percentages = np.logspace(0.3, 1.4, 10) / 100

    # 30% - 100% 
    subset_percentages = np.concatenate((
        subset_percentages,
        np.arange(0.3, 1, 0.1)
    ))

    # Scale to percentages and apply if needed
    subset_percentages = (100 * np.sort(np.round(subset_percentages, 2))).astype(int) if p.subsets else None

    subset_percentages = subset_percentages[subset_percentages == 11]

    MODEL_CONFIGS = create_model_variations(MODEL_CONFIGS, subset_percentages)

    #####################################
    ############ Create jobs ############
    #####################################

    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'run_careful_whisper_results.py')
    job_string = f"{DSQ_MODULES.replace('dark_matter', 'prosody')} srun python {script_fn}"
    job_num = 0

    # failed_jobs = [4]

    counter = 0

    for i, (model_name, model_config) in enumerate(MODEL_CONFIGS.items()):

        # This becomes the name of the output file
        dataset_model = f'{p.train_dataset}_{model_name}'
        model_config = ' '.join(model_config)

        if 'audiovisual' not in model_name:
            continue

        # if counter not in failed_jobs:
        #     counter += 1
        #     continue
        
        # counter += 1

        for split in splits:

            print (dataset_model)
            
            if p.test_dataset == 'pfka-moth-stories':
                out_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', split, 'careful-whisper', dataset_model, f'window-size-{str(WINDOW_SIZE).zfill(5)}')
                out_fn = os.path.join(out_dir, f'task-{split}_window-size-{str(WINDOW_SIZE).zfill(5)}_top-{TOP_N}.csv')

                file_exists = os.path.exists(out_fn)
            else:

                results_dir = os.path.join(BASE_DIR, f'derivatives/careful-whisper/{p.test_dataset}/')
                results_dir = f'{results_dir}m/'if p.test_dataset == 'gigaspeech' else results_dir
                out_fn = os.path.join(results_dir, f'{dataset_model}_test.csv')

                file_exists = os.path.exists(out_fn)

            if file_exists and not p.overwrite:
                continue
            
            ckpt_path = sorted(glob.glob(os.path.join(CKPTS_DIR, model_name, 'checkpoints', 'epoch*.ckpt')))[-1]

            cfg = (
                f"experiment={EXPERIMENT_NAME}.yaml "
                f"{model_config} "
                f"{dataset_config} "
            )
    
            cmd = (
                f"{job_string} "
                f"-d {p.test_dataset} "
                f"-s {split} "
                f"-model_name {dataset_model} "
                f"-ckpt_path {ckpt_path} "
                f"-overrides {cfg} "
            )

            if 'audiovisual-careful-whisper' in model_name:
                token_fusion_ckpt = os.path.join(MODELS_DIR, 'logs/train/token-fusion', f"{p.train_dataset}/{utils.DATASET_CONFIG[p.train_dataset]['ckpt_path']}")

                cmd += (
                    f"-token_fusion_ckpt {token_fusion_ckpt}"
                )

            all_cmds.append(cmd)
            job_num += 1

    dsq_base_string = f'dsq_{p.train_dataset}_careful_whisper_model_results'
    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/behavioral/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/joblists')

    attempt_makedirs(logs_dir)
    attempt_makedirs(dsq_dir)
    attempt_makedirs(joblists_dir)

    joblist_fn = os.path.join(joblists_dir, f'{p.train_dataset}_run_careful_whisper_results.txt')

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
        f"--time={TIME} --account={ACCOUNT} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--exclude={EXCLUDE} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)