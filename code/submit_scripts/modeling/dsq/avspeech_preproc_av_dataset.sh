#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/avspeech_preproc_av_dataset/avspeech_preproc_av_dataset-%A_%2a-%N.txt
#SBATCH --array 0-34
#SBATCH --job-name dsq-avspeech_preproc_av_dataset
#SBATCH --partition=v100_preemptable --time=2-12:00:00 --nodes=1 --gres=gpu:1 --account=dbic --ntasks-per-node=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G --exclude=

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/avspeech_preproc_av_dataset.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/avspeech_preproc_av_dataset

