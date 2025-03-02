#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_mfa_align_huggingface/dsq_mfa_align_huggingface-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-mfa_align_huggingface_joblist
#SBATCH --partition preemptable --time=5-00:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/mfa_align_huggingface_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_mfa_align_huggingface

