#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_run_cut_participant_stimuli/dsq_run_cut_participant_stimuli-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-pilot-multimodal-01_task-nwp_practice_trial_cut_participant_stimuli
#SBATCH --time=240 --nodes=1 --partition=standard --ntasks-per-node=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/pilot-multimodal-01_task-nwp_practice_trial_cut_participant_stimuli.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_run_cut_participant_stimuli

