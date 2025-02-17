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
import dataset_utils as utils

PARTITION = 'standard'
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
    ]}
}

MODEL_ARGS = {

    # # General GPT2-esque model
    # 'careful-whisper_no-xattn': [
    #     f"model.config.cross_attention=False",
    #     f"model.config.use_causal_cross_attention=False",
    # ],

    # # Whisper w/ CLM integration
    # 'careful-whisper_causal-xattn': [
    #     f"model.config.cross_attention=True",
    #     f"model.config.use_causal_cross_attention=True",

    #     # Add in dropout and position embedding
    #     f"model.config.context_embed_dropout=0.1",
    #     f"model.config.context_pos_embed=True",
    # ],

    # # Whisper w/ CLM integration
    # 'careful-whisper_audio-token-fusion': [
    #     f"model.config.token_fusion=True",
    # ],

    # Whisper w/ CLM integration
    'prosody-whisper_causal-xattn': [
        f"model.config.cross_attention=True",
        f"model.config.use_causal_cross_attention=True",

        # Prosody embedding information
        f"model.config.context_type=prominence",
        f"model.config.context_dim=1",
        f"model.config.context_embed_dropout=0.1",
        f"model.config.context_pos_embed=True",

    ],

    # # Whisper w/ CLM integration
    # 'prosody-whisper_token-fusion': [
    #     f"model.config.token_fusion=True",

    #     # Prosody embedding information
    #     f"model.config.context_type=prominence",
    #     f"model.config.context_dim=1",
    # ],
}

def create_model_variations(base_configs, subset_percentages=None):
    """Create model config variations for different subset sizes."""
    variations = {}
    
    for model_name, config in base_configs.items():
        # Add full dataset version
        if subset_percentages is None:
            variations[model_name] = config.copy()
            continue
        
        # Add subset versions
        for subset in subset_percentages:
            subset_name = f"{model_name}-subset-{subset:.2f}"
            subset_config = config.copy()
            variations[subset_name] = subset_config
            
    return variations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-train', '--train_dataset', type=str)
    parser.add_argument('-test', '--test_dataset', type=str, default=None)
    parser.add_argument('-subsets', '--subsets', type=str, default=None)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    if not p.test_dataset:
        p.test_dataset = p.train_dataset

    print (p.test_dataset)

    MODELS_DIR = os.path.join(BASE_DIR, 'code/modeling/joint-clm-prosody/')
    EXPERIMENT = ["experiment=careful_whisper.yaml"]

    CKPTS_DIR = os.path.join(MODELS_DIR, f'logs/train/careful-whisper/{p.train_dataset}/')

    # grab the tasks
    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'run_careful_whisper_results.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'

    # Sub in the conda environment
    job_string = job_string.replace('dark_matter', 'prosody')
    job_num = 0

    # Set up dataset info as the test dataset
    dataset_config = DATASET_INFO[p.test_dataset]['data_config']
    splits = DATASET_INFO[p.test_dataset]['splits']

    if p.subsets:
        # Log space 2 - 25% 
        subset_percentages = np.logspace(0.3, 1.4, 10) / 100 if p.subsets else []
        
        # 1-10%
        # subset_percentages = np.arange(0.01, 0.1, 0.01) if p.subsets else []
        # print (subset_percentages)

        # # 10-100%
        # subset_percentages = np.arange(0.1, 1, 0.1) if p.subsets else []

        MODEL_ARGS = create_model_variations(MODEL_ARGS, subset_percentages)

        # print (MODEL_ARGS.keys())
        # sys.exit(0)

    # TOP_N = 1
    # WINDOW_SIZE = 25
    counter = 0

    for i, (model_name, cfg_args) in enumerate(MODEL_ARGS.items()):

        dataset_model = f'{p.train_dataset}_{model_name}'

        for split in splits:

            # if counter not in [9]:
            #     counter += 1
            #     continue
            
            # counter += 1

            print (dataset_model)

            # if p.train_dataset == 'peoples-speech' and (counter not in [4,7]):
            #     continue
            # if p.train_dataset == 'tedlium' and (counter not in [8, 10, 11]):
            #     continue
            
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
            
            ckpt_path = glob.glob(os.path.join(CKPTS_DIR, model_name, 'checkpoints', 'epoch*.ckpt'))[-1]

            cfg = EXPERIMENT + dataset_config + cfg_args
            cfg = ' '.join(cfg)

            cmd = ''.join([
                f"{job_string} -d {p.test_dataset} -s {split} -model_name {dataset_model} -ckpt_path {ckpt_path} -overrides {cfg}"
            ])

            all_cmds.append(cmd)
            job_num += 1

    dsq_base_string = f'dsq_{p.train_dataset}_careful_whisper_model_results'
    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/behavioral/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/joblists')

    utils.attempt_makedirs(logs_dir)
    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblists_dir)

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