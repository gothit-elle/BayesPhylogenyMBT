#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-3:00:00

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4
module load anaconda/3-5.3.1

conda create -n "myenv" python=3.10.4
source activate
conda activate myenv
python3 -m venv --copies abdou_env

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# Launch multiple process python code
echo "running code"
time srun -n 8 python simulate_tree.py