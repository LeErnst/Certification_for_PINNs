#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --job-name=fem_sol
#SBATCH --partition=single

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode
python 01e_error_estimation_shape_Opt_box_lb.py > data_01e.dat
