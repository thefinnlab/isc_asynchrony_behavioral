#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/dsq_run_cut_participant_audio/dsq_run_cut_participant_audio-%A_%3a-%N.out
#SBATCH --array 0-166
#SBATCH --job-name dsq-test_large_task-black_cut_participant_audio
#SBATCH --time=240 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/joblists/test_large_task-black_cut_participant_audio.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/dsq_run_cut_participant_audio

