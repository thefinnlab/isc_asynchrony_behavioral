import os

BASE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/'
CACHE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/models/'

DERIVATIVES_DIR = os.path.join(BASE_DIR, 'derivatives')
STIM_DIR = os.path.join(BASE_DIR, 'stimuli')
SUBMIT_DIR = os.path.join(BASE_DIR, 'code', 'submit_scripts')
JOBLIST_DIR = os.path.join(SUBMIT_DIR, 'joblists')
DSQ_DIR = os.path.join(SUBMIT_DIR, 'dsq')
LOGS_DIR = os.path.join(BASE_DIR, 'derivatives', 'logs')

if not os.path.exists(JOBLIST_DIR):
	os.makedirs(JOBLIST_DIR)

# Strings to append at start of run for dSQ
DSQ_MODULES	 = [
    'source /optnfs/common/miniconda3/etc/profile.d/conda.sh',
    'conda activate asynchrony'
]

DSQ_MODULES = ''.join([f'{module}; ' for module in DSQ_MODULES])


## HUGGING FACE CONFIG
from huggingface_hub import login

HF_TOKEN = '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'
login(token=HF_TOKEN)
