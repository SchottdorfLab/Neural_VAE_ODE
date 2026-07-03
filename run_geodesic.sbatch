#!/bin/bash
#SBATCH --job-name=geodesic
#SBATCH --partition=gpu-t4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/geodesic-%j.out
#SBATCH --error=logs/geodesic-%j.err

set -euo pipefail

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/4429/envs/odevae

cd /home/4429/projects/Neural_VAE_ODE
mkdir -p logs

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export GEODESIC_DATASET=e65
export GEODESIC_E65_PATH=src/npz_e65_data/E65_data.npz

python sandbox/Geodesic_dynamics/simulate_geodesic_sphere.py





