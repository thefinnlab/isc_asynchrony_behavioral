#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/huth-moth/dsq_word_substitution/dsq_word_substitution-%A_%1a-%N.txt
#SBATCH --array 0-1
#SBATCH --job-name dsq-huth-moth_word_substitution_joblist
#SBATCH --partition=standard --time=240 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=32G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/huth-moth/joblists/huth-moth_word_substitution_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/huth-moth/dsq_word_substitution

