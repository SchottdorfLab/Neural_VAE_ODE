#!/bin/bash -l
#SBATCH --job-name=geodesic_e65
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/geodesic-e65-%j.out
#SBATCH --error=logs/geodesic-e65-%j.err

cd ~/projects/Neural_VAE_ODE

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

echo "Python:"
which python
python --version

echo "GPU:"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

export MPLBACKEND=Agg
export GEODESIC_EPOCHS=500
export GEODESIC_LATENT_DIM=5
export GEODESIC_HEATMAP_PATH=runs/geodesic_fit_e65/darwin_full_${SLURM_JOB_ID}.png

python -u sandbox/Geodesic_dynamics/fit_E65.py
