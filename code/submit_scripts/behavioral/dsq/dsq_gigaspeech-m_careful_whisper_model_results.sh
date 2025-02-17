#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_gigaspeech-m_careful_whisper_model_results/dsq_gigaspeech-m_careful_whisper_model_results-%A_%2a-%N.txt
#SBATCH --array 0-9
#SBATCH --job-name dsq-gigaspeech-m_run_careful_whisper_results
#SBATCH --partition=standard --time=12:00:00 --account=dbic --nodes=1 --ntasks-per-node=1 --ntasks=1 --exclude=q04 --cpus-per-task=8 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/gigaspeech-m_run_careful_whisper_results.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_gigaspeech-m_careful_whisper_model_results

