#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/voxceleb2_classify_language/voxceleb2_classify_language-%A_%2a-%N.txt
#SBATCH --array 0-9
#SBATCH --job-name dsq-voxceleb2_classify_language
#SBATCH --partition=gpuq --time=2-12:00:00 --nodes=1 --gres=gpu:1 --account=dbic --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G --exclude=

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/modeling/joblists/voxceleb2_classify_language.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/modeling/voxceleb2_classify_language

