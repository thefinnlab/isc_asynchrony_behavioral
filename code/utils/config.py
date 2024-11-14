import os, sys
from huggingface_hub import login

HF_TOKEN = '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'
login(token=HF_TOKEN)

## primary directories
BASE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/'
CACHE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/models/'
ATLAS_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/atlases/'
DATASETS_DIR = '/dartfs/rc/lab/F/FinnLab/datasets/'
SCRATCH_DIR = '/dartfs-hpc/scratch/f003rjw/'

## secondary directories
SUBMIT_DIR = os.path.join(BASE_DIR, 'code', 'submit_scripts')
LOGS_DIR = os.path.join(BASE_DIR, 'derivatives', 'logs')

## SET ATLAS DIRECTORIES
os.environ['NILEARN_DATA'] = os.path.join(ATLAS_DIR, 'nilearn_data')
os.environ['NEUROMAPS_DATA'] =  os.path.join(ATLAS_DIR, 'neuromaps_data')

# SET MODELS DIRECTORIES
os.environ['GENSIM_DATA_DIR'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['TORCH_HOME'] = CACHE_DIR

# Strings to append at start of run for dSQ
DSQ_MODULES	 = [
    'module load cuda/11.2',
    'module load openmpi',
    'module load workbench/1.50',
    'source /optnfs/common/miniconda3/etc/profile.d/conda.sh',
    'conda activate dark_matter'
    # 'conda activate asynchrony'
]

DSQ_MODULES = ''.join([f'{module}; ' for module in DSQ_MODULES])