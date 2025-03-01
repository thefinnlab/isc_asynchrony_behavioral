import sys, os
import argparse
import subprocess
from pathlib import Path
import glob

sys.path.append('../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../')

import utils 

PARTITION = 'standard'
TIME = '4-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 31
MEM_PER_CPU = '4G'
GPU_INFO = ''

if __name__ == "__main__":

  DATASETS = ['lrs3'] #, 'voxceleb2']

  logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
  dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
  joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

  attempt_makedirs(logs_dir)
  attempt_makedirs(dsq_dir)
  attempt_makedirs(joblists_dir)

  all_cmds = []

  prosody_dir = os.path.join(os.getcwd(), 'wavelet_prosody_toolkit/wavelet_prosody_toolkit/')
  script_fn = os.path.join(prosody_dir, 'prosody_labeller.py')
  job_string = f'{DSQ_MODULES.replace('dark_matter', 'prosody')} python {script_fn}'
  job_num = 0

  # splits = ['train', 'validation', 'test']
  config_fn = os.path.join(prosody_dir, f'configs/prosody.yaml')

  for dataset in DATASETS:

    if dataset == 'gigaspeech':
      dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset, 'm')
    else:
      dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)

    splits = utils.DATASET_CONFIGS[dataset]['splits']
    print (f'Splits: {splits}', flush=True)

    for split in splits:

        audio_dir = os.path.join(dataset_dir, 'audio', split)
        annotation_dir = os.path.join(dataset_dir, 'textgrids', split)
        output_dir = os.path.join(dataset_dir, 'prosody', split)

        print(f'Making job for: {dataset}, {split}', flush=True)

        cmd = f'{job_string} {audio_dir} -a {annotation_dir} -c {config_fn} -j {CPUS_PER_TASK} -o {output_dir} '
        all_cmds.append(cmd)
        job_num += 1
  
  if not all_cmds:
    print(f'No matching audio and text files found', flush=True)
    sys.exit(0)

  joblist_fn = os.path.join(joblists_dir, f'prosody_extractor_joblist.txt')

  with open(joblist_fn, 'w') as f:
    for cmd in all_cmds:
      f.write(f"{cmd}\n")

  dsq_base_string = f'dsq_prosody_extractor'
  dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
  dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
  array_fmt_width = len(str(job_num))
  
  if not os.path.exists(dsq_out_dir):
    os.makedirs(dsq_out_dir)

  # subprocess.run('module load dSQ', shell=True)
  subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
    f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
    f"--partition {PARTITION} {GPU_INFO} --time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} "
    f"--ntasks={N_TASKS} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
