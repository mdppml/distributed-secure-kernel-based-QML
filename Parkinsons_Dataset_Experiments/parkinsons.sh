#!/bin/bash
#SBATCH --job-name=parkinsons
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=12
#SBATCH --time=15-23:00:00
#SBATCH --partition=month
#SBATCH --mem=240000

echo "Starting job script"
eval "$(conda shell.bash hook)"

echo "Activating conda environment"
conda activate /miniconda3/envs/private_quantum_kernel

if [[ $? -ne 0 ]]; then
  echo "Failed to activate conda environment"
  exit 1
fi

mpirun -np 16 python -u parkinsons.py
