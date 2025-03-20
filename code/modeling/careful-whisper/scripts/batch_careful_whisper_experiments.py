import sys, os
import glob
import argparse
import subprocess
import numpy as np

from config import *
import utils

PARTITION='preemptable'
TIME = '24:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '16:00:00'
EXCLUDE = ''#'p02,p03'
CPUS_PER_TASK = 16
MEM_PER_CPU = '8G'
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'dbic' #dbic

DATASET_INFO = {
    'gigaspeech-m': [
        'data.dataset_name=gigaspeech-m',
        'data.data_dir=\${paths.data_dir}/gigaspeech/m',
        'data.cache_dir=\${paths.cache_dir}/nlp-datasets/gigaspeech/m',
    ],
    'libritts-r': [
        'data.dataset_name=libritts-r',
        'data.data_dir=\${paths.data_dir}/libritts-r',
        'data.cache_dir=\${paths.cache_dir}/nlp-datasets/libritts-r',
    ],
    'tedlium': [
        'data.dataset_name=tedlium',
        'data.data_dir=\${paths.data_dir}/tedlium',
        'data.cache_dir=\${paths.cache_dir}/nlp-datasets/tedlium',
    ],
    'peoples-speech': [
        'data.dataset_name=peoples-speech',
        'data.data_dir=\${paths.data_dir}/peoples-speech',
        'data.cache_dir=\${paths.cache_dir}/nlp-datasets/peoples-speech',
    ],
    'lrs3': [
        'data.dataset_name=lrs3',
        'data.data_dir=\${paths.data_dir}/lrs3',
    ],
    'voxceleb2': [
        'data.dataset_name=voxceleb2',
        'data.data_dir=\${paths.data_dir}/voxceleb2',
    ],
    'avspeech': [
        'data.dataset_name=avspeech',
        'data.data_dir=\${paths.data_dir}/avspeech',
    ],
    'av-combined': [
        'data.dataset_name=av-combined',
        'data.data_dir=\${paths.data_dir}/av-combined',
    ]
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

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-subsets', '--subsets', type=int, default=0)
    # parser.add_argument('-o', '--overwrite', type=int, default=0)
    
    p = parser.parse_args()

    EXPERIMENT_NAME = 'careful_whisper'
    
    MODEL_CONFIGS = {

        # # General GPT2-esque model
        # 'text-careful-whisper_no-xattn': [
        #     f"model.config.cross_attention=False",
        #     f"model.config.use_causal_cross_attention=False",
        # ],

        # Whisper w/ CLM integration
        'audio-careful-whisper_causal-xattn': [
            f"model.config.cross_attention=True",
            f"model.config.use_causal_cross_attention=True",

            # Add in dropout and position embedding
            f"model.config.context_embed_dropout=0.1",
            f"model.config.context_pos_embed=True",
        ],

        # # Whisper w/ CLM integration
        # 'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp': [
        #     f"model.config.cross_attention=True",
        #     f"model.config.use_causal_cross_attention=True",

        #     # Prosody embedding information
        #     f"model.config.context_type=audiovisual_features",
        #     f"model.config.context_embed_dropout=0.1",
        #     f"model.config.context_pos_embed=True",
        #     # f"model.optimizer.lr=5e-5",

        #     # How to fusion AV tokens
        #     f'data.token_fusion_method=mlp',
        #     "data.ckpt_path=\${paths.log_dir}train/token-fusion/" + f"{p.dataset}/{utils.DATASET_CONFIG[p.dataset]['ckpt_path']}"
        # ],

        # # Whisper w/ CLM integration
        # 'prosody-careful-whisper_causal-xattn': [
        #     f"model.config.cross_attention=True",
        #     f"model.config.use_causal_cross_attention=True",

        #     # Prosody embedding information
        #     f"model.config.context_type=prominence",
        #     f"model.config.context_dim=1",
        #     f"model.config.context_embed_dropout=0.1",
        #     f"model.config.context_pos_embed=True",

        # ],

        # # Whisper w/ CLM integration
        # 'visual-careful-whisper_causal-xattn': [
        #     f"model.config.cross_attention=True",
        #     f"model.config.use_causal_cross_attention=True",

        #     # Prosody embedding information
        #     f"model.config.context_type=video_features",
        #     f"model.config.context_embed_dropout=0.1",
        #     f"model.config.context_pos_embed=True",

        # ],

        # # Whisper w/ CLM integration
        # 'careful-whisper_audio-token-fusion': [
        #     f"model.config.token_fusion=True",
        # ],

        # # Whisper w/ CLM integration
        # 'prosody-whisper_token-fusion': [
        #     f"model.config.token_fusion=True",

        #     # Prosody embedding information
        #     f"model.config.context_type=prominence",
        #     f"model.config.context_dim=1",
        # ],


        # # Whisper w/ CLM integration
        # 'careful-whisper_audio-xattn_shuffled': [
        # 	f"model.config.shuffle_context=True",
        # 	f"model.config.cross_attention=True",
        # 	f"model.config.use_causal_cross_attention=True",
        # ],


        # # Whisper w/ CLM integration
        # 'careful-whisper_audio-embed-only': [
        # 	f"model.config.embed_type=audio_inputs",
        # 	f"model.config.cross_attention=False",
        # 	f"model.config.use_causal_cross_attention=False",
        # ]

        # # Switch self-attention to audio paying attention to text
        # 'careful-whisper_causal-xattn_inverse-audio-text': [
        # 	f"model.config.cross_attention=True",
        # 	f"model.config.use_causal_cross_attention=True",

        # 	# Audio information
        # 	f"model.config.context_embed_dropout=0.1",
        # 	f"model.config.context_pos_embed=True",
        # 	f"model.config.inverse_audio_text=True",

        # ],

        # Controlling for the number of model parameters
        # 'careful-whisper_no-xattn_parameter-control': [
        # 	f"model.config.num_layers=18",
        # ],

        # Seeing if bidirectional cross attention works
        # 'careful-whisper_causal-bi-xattn': [
        # 	f"model.config.bidirectional_cross_attention=True",
        # 	f"model.config.use_causal_cross_attention=True",
        # ],

        # # Matched architecture to GPT2 (same embed dim + num_heads)
        # 'careful-whisper_gpt2-control': [
        # 	f"model.config.embed_dim=768",
        # 	f"model.config.num_heads=12",
        # 	f"model.config.cross_attention=False",
        # 	f"model.config.use_causal_cross_attention=False",
        # ],

        # # Whisper w/ CLM integration
        # 'careful-whisper_causal-xattn_num-layers-6': [
        #     f"model.config.num_layers=6",
        #     f"model.config.cross_attention=True",
        #     f"model.config.use_causal_cross_attention=True",

        #     # Add in dropout and position embedding
        #     f"model.config.context_embed_dropout=0.1",
        #     f"model.config.context_pos_embed=True",
        # ],

        # # General GPT2-esque model
        # 'careful-whisper_no-xattn_num-layers-6': [
        #     f"model.config.num_layers=6",
        #     f"model.config.cross_attention=False",
        #     f"model.config.use_causal_cross_attention=False",
        # ],
    }

    # make directories
    dsq_dir = os.path.join(SUBMIT_DIR, 'dsq')
    joblist_dir = os.path.join(SUBMIT_DIR, 'joblists')
    logs_dir = os.path.join(LOGS_DIR)

    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblist_dir)
    utils.attempt_makedirs(logs_dir)

    all_cmds = []
    script_fn = os.path.join(BASE_DIR, 'src/train.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'
    job_num = 0

    # Dataset overrides --> set up dataset
    dataset_config = ' '.join(DATASET_INFO[p.dataset])

    # Define subset percentages and create model variations

    # # Log space 2 - 25% 
    subset_percentages = np.logspace(0.3, 1.4, 10) / 100

    # # 30% - 100% 
    # subset_percentages = np.concatenate((
    #     subset_percentages,
    #     np.arange(0.3, 0.5, 0.1)
    # ))
    subset_percentages = subset_percentages[np.logical_and(subset_percentages > 0.1, subset_percentages < 0.15)]

    # Scale to percentages and apply if needed
    subset_percentages = (100 * np.sort(np.round(subset_percentages, 2))).astype(int) if p.subsets else None

    MODEL_CONFIGS = create_model_variations(MODEL_CONFIGS, subset_percentages)

    wandb_group = 'full-data' if not p.subsets else 'subsets'
    counter = 0

    # failed_jobs = [5, 6, 7]

    for model_name, model_config in MODEL_CONFIGS.items():

        # if (counter not in failed_jobs): # and (counter < 9):
        #     counter += 1
        #     continue

        # counter += 1
        print (model_name)

        # Model config --> change model configurations
        model_config = ' '.join(model_config)
        
        # Logger overrides --> change the name based on the dataset
        wandb_config = (
            f"logger.wandb.project={p.dataset}-audiovisual "
            f"logger.wandb.name={model_name} "
            f"logger.wandb.group={model_name.split('_')[0]} "
            f"logger.wandb.job_type={wandb_group} "
        )

        hydra_config = (
            "hydra.run.dir=\${paths.log_dir}/\${task_name}/careful-whisper/"
            f"{p.dataset}/"
            f"{model_name}/"
        )

        cmd = (
            f"{job_string} "
            f"experiment={EXPERIMENT_NAME}.yaml "
            f"{model_config} "
            f"{dataset_config} "
            f"{wandb_config} "
            f"{hydra_config} "
        )

        all_cmds.append(cmd)
        job_num += 1

    if not all_cmds:
        print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_careful_whisper_experiments.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    
    dsq_base_string = f'dsq_{p.dataset}_careful_whisper_experiments'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num)) 

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)
    
    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --account={ACCOUNT} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--exclude={EXCLUDE} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
