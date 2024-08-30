#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/huth-moth/dsq_plot_encoding/dsq_plot_encoding-%A_%1a-%N.txt
#SBATCH --array 0-2
#SBATCH --job-name dsq-huth-moth_plot_encoding_joblist
#SBATCH --partition=standard --time=10:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=16G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/submit_scripts/huth-moth/joblists/huth-moth_plot_encoding_joblist.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/huth-moth/dsq_plot_encoding

