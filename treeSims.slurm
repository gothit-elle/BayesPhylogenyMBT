#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=4-24:00:00

# Load required modules
module purge
module load foss/2022a
module load Python
module load mpi4py/3.1.4

python3 -m venv myenv
source myenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Launch multiple process python code
echo "running code"
srun python3 simulate_tree.py