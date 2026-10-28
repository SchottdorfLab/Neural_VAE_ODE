#!/bin/bash -l
#SBATCH --job-name=jax_e65_ctx
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/jax-e65-context-%A_%a.out
#SBATCH --error=logs/jax-e65-context-%A_%a.err

set -euo pipefail

cd "$HOME/projects/Neural_VAE_ODE"
mkdir -p logs runs

PYTHON_BIN="$HOME/envs/odevae/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN" >&2
    exit 1
fi

CONTEXT_VALUES=(2 5 10 20)
CONTEXT_STEPS="${CONTEXT_VALUES[$SLURM_ARRAY_TASK_ID]}"
MAX_TRIALS="${JAX_E65_MAX_TRIALS:-60}"
EPOCHS="${JAX_E65_EPOCHS:-1000}"
OUT_DIR="runs/jax_e65_context_scan_n${MAX_TRIALS}_c${CONTEXT_STEPS}_e${EPOCHS}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

export JAX_E65_DATA_PATH="${JAX_E65_DATA_PATH:-src/npz_e65_data/E65_data.npz}"
export JAX_E65_OUT_DIR="$OUT_DIR"
export JAX_E65_SEED="${JAX_E65_SEED:-42}"
export JAX_E65_LATENT_DIM="${JAX_E65_LATENT_DIM:-5}"
export JAX_E65_CONTEXT_STEPS="$CONTEXT_STEPS"
export JAX_E65_MAX_TRIALS="$MAX_TRIALS"
export JAX_E65_HELDOUT_FRAC="${JAX_E65_HELDOUT_FRAC:-0.20}"
export JAX_E65_EPOCHS="$EPOCHS"
export JAX_E65_BATCH_SIZE="${JAX_E65_BATCH_SIZE:-8}"
export JAX_E65_LR="${JAX_E65_LR:-0.001}"
export JAX_E65_LOG_EVERY="${JAX_E65_LOG_EVERY:-25}"
export JAX_E65_CHECKPOINT_EVERY="${JAX_E65_CHECKPOINT_EVERY:-100}"
export JAX_E65_RESUME="${JAX_E65_RESUME:-false}"
export JAX_E65_X64="${JAX_E65_X64:-true}"
export JAX_E65_REQUIRE_GPU=true

echo "Context scan task: ${SLURM_ARRAY_TASK_ID}"
echo "Context frames: ${CONTEXT_STEPS}"
echo "Trials: ${MAX_TRIALS} (fixed seed ${JAX_E65_SEED})"
echo "Epochs: ${EPOCHS}"
echo "Output: ${OUT_DIR}"
echo "Host: $(hostname)"
echo "Started: $(date)"

"$PYTHON_BIN" - <<'PY'
import jax
print("JAX devices:", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX cannot see the allocated GPU")
PY

"$PYTHON_BIN" -u scripts/run_tmp_e65_notebook.py

echo "Finished: $(date)"
