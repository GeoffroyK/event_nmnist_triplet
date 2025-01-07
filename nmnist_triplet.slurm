#!/bin/bash
#SBATCH --job-name=nmnist_triplet   # Job name
#SBATCH --output=nmnist_%j.log      # Output file name (%j expands to jobID)
#SBATCH --error=nmnist_%j.err       # Error file name (%j expands to jobID)
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --mem=16G                   # Memory per node
#SBATCH --time=24:00:00            # Time limit hrs:min:sec
#SBATCH --gres=gpu:1               # Request 1 GPU

# Activate conda environment
conda activate eventvpr

# Run your training script
python train_network.py 