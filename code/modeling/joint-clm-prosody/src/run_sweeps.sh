#!/bin/bash

# Name of the job
#SBATCH --job-name=gigaspeech_sweeps
#SBATCH -o gigaspeech-sweeps_%A.txt

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=8

# Request memory
#SBATCH --mem=8G

# Walltime (job duration)
#SBATCH --time=3-00:00:00

# GPU INFO
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

module load cuda/12.3
module load openmpi

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate prosody

echo "Cuda devices: " ${CUDA_VISIBLE_DEVICES} 

wandb agent finnlab/isc_asynchrony_behavior-code_modeling_joint-clm-prosody/06tgn7zj