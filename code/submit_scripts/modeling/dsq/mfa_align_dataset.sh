#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/mfa_align_dataset/mfa_align_dataset-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-mfa_align_dataset_joblist
#SBATCH --partition preemptable --time=5-00:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=31 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/mfa_align_dataset_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/mfa_align_dataset

