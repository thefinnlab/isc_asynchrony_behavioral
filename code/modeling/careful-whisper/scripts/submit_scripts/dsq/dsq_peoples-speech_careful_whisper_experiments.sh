#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/logs/dsq_peoples-speech_careful_whisper_experiments/dsq_peoples-speech_careful_whisper_experiments-%A_%1a-%N.txt
#SBATCH --array 0-4
#SBATCH --job-name dsq-peoples-speech_careful_whisper_experiments
#SBATCH --partition=v100_preemptable --time=2-12:00:00 --account=dbic --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/joblists/peoples-speech_careful_whisper_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/joint-clm-prosody/scripts/submit_scripts/logs/dsq_peoples-speech_careful_whisper_experiments

