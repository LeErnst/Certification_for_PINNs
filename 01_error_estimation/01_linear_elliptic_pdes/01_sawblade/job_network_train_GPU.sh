#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=06:00:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH -J 01a_network_train_GPU

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode_cuda
python 01a_network_training_sawblade.py

