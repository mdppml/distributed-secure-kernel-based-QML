#!/bin/bash
#SBATCH --job-name=heart
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=12
#SBATCH --time=22-23:14:00
#SBATCH --partition=month
#SBATCH --output=heart_dataset_final_run_%j.txt
#SBATCH --mem=240000

echo "Starting job script"
eval "$(conda shell.bash hook)"

echo "Activating conda environment"
conda activate /miniconda3/envs/private_quantum_kernel


if [[ $? -ne 0 ]]; then
  echo "Failed to activate conda environment"
  exit 1
fi

mpirun -np 16 python -u heart.py 
