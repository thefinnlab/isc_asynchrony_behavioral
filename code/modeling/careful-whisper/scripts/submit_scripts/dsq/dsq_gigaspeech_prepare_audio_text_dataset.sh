#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/logs/dsq_gigaspeech_prepare_audio_text_dataset/dsq_gigaspeech_prepare_audio_text_dataset-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-gigaspeech_prepare_audio_text_dataset
#SBATCH --partition=gpuq --time=2-23:00:00 --account=dbic --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=16G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/joblists/gigaspeech_prepare_audio_text_dataset.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/logs/dsq_gigaspeech_prepare_audio_text_dataset

