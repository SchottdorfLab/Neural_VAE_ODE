#!/bin/bash -l
#SBATCH --job-name=e65_geo
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=tesla_t4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/e65-trials-%j.out
#SBATCH --error=logs/e65-trials-%j.err

cd ~/projects/Neural_VAE_ODE

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export E65_DEVICE=cuda
export E65_DTYPE=float32
export E65_NPZ_PATH=src/npz_e65_data/E65_data.npz

# Keep this path fixed so replacement jobs find the checkpoints.
export E65_OUT_DIR=runs/geodesic_e65_trials_5d_e300
export E65_CHECKPOINT_EVERY=1
export E65_RESUME=true

export E65_SEED=42
export E65_LATENT_DIM=5
export E65_EPOCHS=300
export E65_LR=0.001

export E65_HELDOUT_FRAC=0.20
export E65_CONTEXT_STEPS=10
export E65_CONTEXT_EPOCHS=300
export E65_CONTEXT_LR=0.001

export E65_BATCH_SIZE=16
export E65_MAX_TRIALS=0
export E65_ACTIVE_THRESHOLD=0.0
export E65_FILTER_TRAIN_SILENT=true
export E65_MODEL_SOLVER=rk4
export E65_LOG_EVERY=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version

echo "GPU status:"
nvidia-smi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -u sandbox/Geodesic_dynamics/fit_E65_trials.py

echo "Finished: $(date)"
