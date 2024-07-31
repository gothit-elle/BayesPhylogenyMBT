#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-3:00:00

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4

# Launch multiple process python code
echo "running code"
time srun -n 8 python3 simulate_tree.py