#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task 72
#SBATCH --time=7-24:00:00

# Load required modules
module purge
module load foss/2022a
module load Python/3.9
module load mpi4py/3.1.4

python3 -m venv myenv
source myenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python3 -m pip install PyQt5

# Launch multiple process python code
echo "running code"
python simulate_tree.py