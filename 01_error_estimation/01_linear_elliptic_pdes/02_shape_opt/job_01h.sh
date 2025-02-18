#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --mem=16gb
#SBATCH --job-name=lb_sol_time
#SBATCH --partition=single

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode
python 01h_times_error_est_box_lb.py
