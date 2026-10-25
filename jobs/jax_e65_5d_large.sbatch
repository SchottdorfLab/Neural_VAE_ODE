#!/bin/bash -l
#SBATCH --job-name=jax_e65_5d
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/jax-e65-5d-%j.out
#SBATCH --error=logs/jax-e65-5d-%j.err

echo "Batch shell started: $(date)"
echo "Host: $(hostname)"
echo "Working directory before setup: $(pwd)"

cd ~/projects/Neural_VAE_ODE || {
    echo "Could not enter ~/projects/Neural_VAE_ODE" >&2
    exit 1
}
mkdir -p logs runs/jax_tmp_e65_5d_large

# Calling the environment's interpreter directly is more reliable in a
# non-interactive SLURM shell than relying on shell-function activation.
PYTHON_BIN="$HOME/envs/odevae/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export JAX_E65_DATA_PATH="${JAX_E65_DATA_PATH:-src/npz_e65_data/E65_data.npz}"
export JAX_E65_OUT_DIR="${JAX_E65_OUT_DIR:-runs/jax_tmp_e65_5d_large}"
export JAX_E65_SEED="${JAX_E65_SEED:-42}"
export JAX_E65_LATENT_DIM="${JAX_E65_LATENT_DIM:-5}"
export JAX_E65_CONTEXT_STEPS="${JAX_E65_CONTEXT_STEPS:-10}"
export JAX_E65_HELDOUT_FRAC="${JAX_E65_HELDOUT_FRAC:-0.20}"
export JAX_E65_EPOCHS="${JAX_E65_EPOCHS:-3000}"
export JAX_E65_BATCH_SIZE="${JAX_E65_BATCH_SIZE:-8}"
export JAX_E65_LR="${JAX_E65_LR:-0.001}"
export JAX_E65_LOG_EVERY="${JAX_E65_LOG_EVERY:-10}"
export JAX_E65_CHECKPOINT_EVERY="${JAX_E65_CHECKPOINT_EVERY:-25}"
export JAX_E65_RESUME="${JAX_E65_RESUME:-true}"
export JAX_E65_X64="${JAX_E65_X64:-true}"
export JAX_E65_REQUIRE_GPU="${JAX_E65_REQUIRE_GPU:-true}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: ${PYTHON_BIN}"
"$PYTHON_BIN" --version
nvidia-smi
"$PYTHON_BIN" -c "import jax; print('JAX:', jax.__version__); print('Devices:', jax.devices())" || {
    echo "JAX startup check failed." >&2
    exit 1
}

"$PYTHON_BIN" -u scripts/run_tmp_e65_notebook.py
run_status=$?

echo "Finished: $(date), exit status: ${run_status}"
exit "${run_status}"
