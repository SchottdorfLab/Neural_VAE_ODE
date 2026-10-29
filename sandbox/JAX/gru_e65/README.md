# GRU E65 Initial-Condition Model

This experiment replaces the flattened context encoder with a gated recurrent
unit (GRU). The geodesic dynamics, free-dynamics comparison, RK4 solver,
decoder, loss, trial split, and forecast evaluation remain unchanged.

For context frames

\[
x_0,x_1,\ldots,x_{C-1},
\]

the GRU updates a hidden state in temporal order:

\[
h_t=\operatorname{GRU}(x_t,h_{t-1}).
\]

The final hidden state is mapped to the initial latent position and velocity:

\[
(z_0,v_0)=W_h h_{C-1}+b_h.
\]

The model then performs the same autonomous geodesic or free-dynamics rollout
and decodes each latent position into predicted neural activity.

Unlike the flattened encoder, the GRU shares its weights over time. Its
parameter count therefore does not increase when the number of context frames
increases.

## Local smoke test

From the repository root:

```bash
MPLBACKEND=Agg \
JAX_E65_OUT_DIR=runs/jax_e65_gru_smoke \
JAX_E65_MAX_TRIALS=10 \
JAX_E65_CONTEXT_STEPS=5 \
JAX_E65_EPOCHS=1 \
JAX_E65_BATCH_SIZE=4 \
JAX_E65_X64=false \
JAX_E65_REQUIRE_GPU=false \
JAX_E65_RESUME=false \
python -u sandbox/JAX/gru_e65/run_gru_e65.py
```

## Main configuration variables

| Variable | Default | Meaning |
|---|---:|---|
| `JAX_E65_GRU_HIDDEN_DIM` | 128 | GRU hidden-state dimension |
| `JAX_E65_CONTEXT_STEPS` | 10 | Observed prefix frames |
| `JAX_E65_LATENT_DIM` | 5 | Dimensions of both \(z\) and \(v\) |
| `JAX_E65_MAX_TRIALS` | 0 | Trial limit; zero uses all trials |
| `JAX_E65_HELDOUT_FRAC` | 0.20 | Fraction of whole trials held out |
| `JAX_E65_EPOCHS` | 2000 | Training epochs per model |
| `JAX_E65_OUT_DIR` | `runs/jax_tmp_e65_geodesic_vs_free` | Output directory |

Use a unique output directory for each configuration so checkpoints from
different context lengths or encoder settings are never mixed.

## Darwin context-length scan

The included SLURM array tests 2, 5, 10, and 20 context frames. Each task uses
the same seed, 60-trial subset, 48/12 whole-trial split, and 1,000 epochs.

```bash
JOB_ID=$(sbatch --parsable sandbox/JAX/gru_e65/jax_e65_gru_context_scan.sbatch)
JOB_ID=${JOB_ID%%;*}
echo "$JOB_ID"
```

Monitor the four tasks:

```bash
squeue -j "$JOB_ID"
tail -F logs/jax-e65-gru-${JOB_ID}_*.out
```

After all four tasks finish:

```bash
python sandbox/JAX/gru_e65/summarize_context_scan.py
```
