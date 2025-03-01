#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_prosody_extractor/dsq_prosody_extractor-%A_%1a-%N.txt
#SBATCH --array 0-2
#SBATCH --job-name dsq-prosody_extractor_joblist
#SBATCH --partition standard --time=4-00:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=31 --mem-per-cpu=4G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/prosody_extractor_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/dsq_prosody_extractor

