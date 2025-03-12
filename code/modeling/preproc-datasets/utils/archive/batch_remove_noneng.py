import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob

sys.path.append('../../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../../utils/')

import utils 

PARTITION = 'standard'
TIME = '5-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

NODE_LIST = ''#--nodelist=a03,a04'
EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == "__main__":

    dataset = 'voxceleb2'

    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

    attempt_makedirs(logs_dir)
    attempt_makedirs(dsq_dir)
    attempt_makedirs(joblists_dir)

    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'voxceleb2_remove_noneng.py')
    job_num = 0

    dataset_config = utils.DATASET_CONFIGS[dataset]
    splits = dataset_config['splits']
    splits = splits[::-1]

    output_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)

    models = utils.DATASET_TYPES['video']
    models['video_model'] = 'data2vec'

    model_types = ' '.join([f"--{k} {v} " for k, v in models.items()])

    N_SHARDS = 200

    for shard in range(N_SHARDS):
        cmd = [
            f"{DSQ_MODULES.replace('dark_matter', 'prosody')} ",
            f"python voxceleb2_remove_noneng.py -d {dataset} --output_dir {output_dir} {model_types} --num_jobs={CPUS_PER_TASK} --current_shard {shard} --num_shards {N_SHARDS}", 
          ]

        cmd = "".join(cmd)
        all_cmds.append(cmd)
        job_num += 1

    if not all_cmds:
        print(f'No matching audio and text files found', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblists_dir, f'dsq_remove_noneng.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
          f.write(f"{cmd}\n")

    dsq_base_string = f'remove_noneng_files'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)

    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)
