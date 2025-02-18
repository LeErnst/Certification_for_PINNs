#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --job-name=fem_sol
#SBATCH --partition=single

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode
python 010_high_fidelity_sol.py
