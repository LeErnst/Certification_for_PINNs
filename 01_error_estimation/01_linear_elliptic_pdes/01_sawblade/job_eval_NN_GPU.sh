#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=00:30:00
#SBATCH --mem=6gb
#SBATCH --gres=gpu:1
#SBATCH -J 01a_NN_eval_GPU

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode_cuda
python 01e_PINN_evaluation_time.py

