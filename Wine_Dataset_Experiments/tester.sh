#!/bin/bash
#SBATCH --job-name=park_kernel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:14:00
#SBATCH --partition=test
#SBATCH --output=wine_dataset_kernel_%j.txt
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

python -u tester.py 
