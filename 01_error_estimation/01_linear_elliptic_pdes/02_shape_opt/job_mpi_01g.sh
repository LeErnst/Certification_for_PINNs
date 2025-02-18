#!/bin/bash
#SBATCH --job-name=data01c 	    # Job name
#SBATCH --ntasks=40                 # Number of MPI tasks
#SBATCH --time=00:30:00             # Time limit hrs:min:sec
#SBATCH --partition=single          # Partition (queue) name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --cpus-per-task=1           # CPUs per task
#SBATCH --mem=16gb                  # Total memory for the job

# Activate FEniCSx environment (if not using a module)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate disscode

# Run the FEniCSx script with mpirun
mpirun --bind-to core --map-by core python 01g_times_error_est_box_ub.py

