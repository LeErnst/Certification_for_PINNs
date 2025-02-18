#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=36:00:00
#SBATCH --mem=32gb
#SBATCH --partition=single
#SBATCH -J 01a_network_train_multithread

#Usually you should set
export KMP_AFFINITY=compact,1,0

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode
python 01a_network_training_sawblade.py

