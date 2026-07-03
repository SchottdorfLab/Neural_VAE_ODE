#!/bin/bash
#SBATCH --job-name=fit-e65-geo
#SBATCH --partition=gpu-t4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fit-e65-geo-%j.out
#SBATCH --error=logs/fit-e65-geo-%j.err

set -euo pipefail

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/4429/envs/odevae

cd /home/4429/projects/Neural_VAE_ODE
mkdir -p logs runs/geodesic_fit_e65

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export E65_MAT_PATH=src/mat_E65_data/E65.mat
export GEODESIC_LATENT_DIM=5
export GEODESIC_EPOCHS=500
export GEODESIC_HEATMAP_PATH=runs/geodesic_fit_e65/fit_e65_geodesic_heatmap_${SLURM_JOB_ID}.png

python sandbox/Geodesic_dynamics/fit_E65.py
