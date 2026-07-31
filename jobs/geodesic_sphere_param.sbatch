#!/bin/bash -l
#SBATCH --job-name=geo_sphere
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ~/projects/Neural_VAE_ODE

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

export MPLBACKEND=Agg
export GEODESIC_DEVICE=cuda
export SPHERE_LOG_EVERY=${SPHERE_LOG_EVERY:-10}

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python -u sandbox/Geodesic_dynamics/fit_sphere_trials.py
