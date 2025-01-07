#!/bin/bash

#SBATCH --job-name=nmnist_triplet   # Job name
#SBATCH	--partition=gpu 			### Selection of the partition (default, gpu)
#SBATCH --output=nmnist_%j.log      # Output file name (%j expands to jobID)
#SBATCH --error=nmnist_%j.err       # Error file name (%j expands to jobID)
#SBATCH	--nodes=1				### Number of nodes
#SBATCH	--ntasks=1				### Number of tasks
#SBATCH	--gres=gpu:1				### Number of GPUs : 1 GPU
#SBATCH --mem=50G


# Activate conda environment
conda activate eventvpr

# Run your training script
python train_network.py 