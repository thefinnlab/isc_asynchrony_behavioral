#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_prosody_model_results/dsq_prosody_model_results-%A_%1a-%N.txt
#SBATCH --array 0-4
#SBATCH --job-name dsq-run_prosody_model_results
#SBATCH --partition=standard --time=12:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/run_prosody_model_results.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_prosody_model_results

