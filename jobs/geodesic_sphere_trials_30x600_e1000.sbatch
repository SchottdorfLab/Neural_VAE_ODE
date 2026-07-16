#!/bin/bash -l
#SBATCH --job-name=geo_sphere_30
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/geo-sphere-30x600-e1000-%j.out
#SBATCH --error=logs/geo-sphere-30x600-e1000-%j.err

cd ~/projects/Neural_VAE_ODE

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

export MPLBACKEND=Agg
export GEODESIC_DEVICE=cuda
export SPHERE_OUT_DIR=runs/geodesic_sphere_trials_30x600_e1000
export SPHERE_N_TRIALS=30
export SPHERE_N_TIME=600
export SPHERE_T_MAX=12.566370614359172
export SPHERE_N_NEURONS=300
export SPHERE_SPEED=0.70710678
export SPHERE_KAPPA=1.5
export SPHERE_EPOCHS=1000
export SPHERE_LOG_EVERY=10

python -u sandbox/Geodesic_dynamics/fit_sphere_trials.py
