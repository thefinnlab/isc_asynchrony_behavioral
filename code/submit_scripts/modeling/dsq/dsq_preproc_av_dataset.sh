#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_preproc_av_dataset/dsq_preproc_av_dataset-%A_%1a-%N.txt
#SBATCH --array 0-6
#SBATCH --job-name dsq-dsq_preproc_av_dataset
#SBATCH --partition=preemptable --time=5-00:00:00 --nodes=1 --account=dbic --ntasks-per-node=1 --ntasks=1 --cpus-per-task=31 --mem-per-cpu=4G --exclude=

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/dsq_preproc_av_dataset.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_preproc_av_dataset

