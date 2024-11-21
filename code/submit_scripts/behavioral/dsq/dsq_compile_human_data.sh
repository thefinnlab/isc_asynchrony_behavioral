#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_compile_human_data/dsq_compile_human_data-%A_%1a-%N.txt
#SBATCH --array 0-2
#SBATCH --job-name dsq-run_compile_human_data
#SBATCH --partition=standard --time=12:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/behavioral/joblists/run_compile_human_data.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/behavioral/dsq_compile_human_data

