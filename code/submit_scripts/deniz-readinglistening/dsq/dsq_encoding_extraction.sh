#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/deniz-readinglistening/dsq_encoding_extraction/dsq_encoding_extraction-%A_%2a-%N.txt
#SBATCH --array 0-21
#SBATCH --job-name dsq-deniz-readinglistening_encoding_extraction_joblist
#SBATCH --partition=standard --time=240 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=16G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/deniz-readinglistening/joblists/deniz-readinglistening_encoding_extraction_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/deniz-readinglistening/dsq_encoding_extraction

