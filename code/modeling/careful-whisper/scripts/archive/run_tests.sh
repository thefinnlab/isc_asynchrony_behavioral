#!/bin/bash

# Name of the job
#SBATCH --job-name=pytest_joint_clm_prosody
#SBATCH -o pytest_joint_clm_prosody%A.txt

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=8

# Request memory
#SBATCH --mem=8G

# Walltime (job duration)
#SBATCH --time=24:00:00

# GPU INFO
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate prosody 

cd ../

# run pytest on all
pytest tests/test_configs.py

pytest tests/test_datamodules.py

pytest tests/test_metrics.py

pytest tests/test_sweeps.py

pytest tests/test_train.py

