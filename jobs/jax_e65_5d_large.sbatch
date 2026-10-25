#!/bin/bash -l
#SBATCH --job-name=jax_e65_5d
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=tesla_t4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/jax-e65-5d-%j.out
#SBATCH --error=logs/jax-e65-5d-%j.err

set -euo pipefail

cd ~/projects/Neural_VAE_ODE
mkdir -p logs runs/jax_tmp_e65_5d_large

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export JAX_E65_DATA_PATH=src/npz_e65_data/E65_data.npz
export JAX_E65_OUT_DIR=runs/jax_tmp_e65_5d_large
export JAX_E65_SEED=42
export JAX_E65_LATENT_DIM=5
export JAX_E65_CONTEXT_STEPS=10
export JAX_E65_HELDOUT_FRAC=0.20
export JAX_E65_EPOCHS=3000
export JAX_E65_BATCH_SIZE=8
export JAX_E65_LR=0.001
export JAX_E65_LOG_EVERY=10
export JAX_E65_CHECKPOINT_EVERY=25
export JAX_E65_RESUME=true
export JAX_E65_X64=true
export JAX_E65_REQUIRE_GPU=true

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import jax; print('JAX:', jax.__version__); print('Devices:', jax.devices())"

python -u scripts/run_tmp_e65_notebook.py

echo "Finished: $(date)"
