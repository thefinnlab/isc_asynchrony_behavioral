#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/deniz-readinglistening/dsq_timepoint_classification/dsq_timepoint_classification-%A_%3a-%N.txt
#SBATCH --array 0-359
#SBATCH --job-name dsq-deniz-readinglistening_timepoint_classification_joblist
#SBATCH --partition=preemptable --time=24:00:00 --account=dbic --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/deniz-readinglistening/joblists/deniz-readinglistening_timepoint_classification_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/deniz-readinglistening/dsq_timepoint_classification

