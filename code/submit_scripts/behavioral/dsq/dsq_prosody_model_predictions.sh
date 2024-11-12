#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_prosody_model_predictions/dsq_prosody_model_predictions-%A_%2a-%N.txt
#SBATCH --array 0-1
#SBATCH --job-name dsq-run_prosody_model_predictions
#SBATCH --partition=standard --time=12:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/run_prosody_model_predictions.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_prosody_model_predictions

