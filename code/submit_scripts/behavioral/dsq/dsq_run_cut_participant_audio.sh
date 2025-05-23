#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_run_cut_participant_audio/dsq_run_cut_participant_audio-%A_%3a-%N.out
#SBATCH --array 0-149
#SBATCH --job-name dsq-final-multimodal-01_task-demon_cut_participant_audio
#SBATCH --time=240 --nodes=1 --partition=standard --ntasks-per-node=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH --exclude=s13,s24,s28

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/final-multimodal-01_task-demon_cut_participant_audio.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_run_cut_participant_audio

