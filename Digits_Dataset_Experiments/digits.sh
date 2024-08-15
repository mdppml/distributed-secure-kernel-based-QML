#!/bin/bash
#SBATCH --job-name=digits
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:14:59
#SBATCH --partition=test
#SBATCH --output=testjob_%j.txt
#SBATCH --mem=240000

echo "Starting job script"
eval "$(conda shell.bash hook)"

# Activate the Conda environment
echo "Activating conda environment"
conda activate /miniconda3/envs/private_quantum_kernel


if [[ $? -ne 0 ]]; then
  echo "Failed to activate conda environment"
  exit 1
fi

mpirun -np 4 python -u digits.py 
