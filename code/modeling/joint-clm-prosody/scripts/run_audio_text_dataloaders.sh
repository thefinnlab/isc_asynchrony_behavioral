#!/bin/bash

# Name of the job
#SBATCH --job-name=prosody_expt
#SBATCH -o prosody_expt-%A.txt

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=8

# Request memory
#SBATCH --mem=8G

# Walltime (job duration)
#SBATCH --time=1-00:00:00

# GPU INFO
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

# get tunneling info
node=$(hostname -s)
user=$(whoami)
cluster="discovery8"

# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

module load cuda/12.3
module load openmpi

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate prosody

echo "Cuda devices: " ${CUDA_VISIBLE_DEVICES}

cd ../

python src/train.py -m experiment=joint_clm_prosody.yaml model.loss_mode="clm"

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

