# Best recorded JAX sphere training reconstruction

This record preserves the results embedded in
`sandbox/JAX/tmp_fit_sphere_trials_comparison.ipynb`.

| Split | R² | r |
|---|---:|---:|
| Training reconstruction | 0.925482 | 0.962150 |
| Held-out forecast | -0.451123 | 0.256559 |

The model used 24 training trials and 6 completely held-out trials, with 600
timepoints and 300 simulated neurons per trial. It inferred latent initial
position and velocity from the first two neural frames, then generated the
remaining trajectory using a learned 2D geodesic model and RK4 integration.

The saved figure shows the first held-out trial (original dataset trial 17).
The neural heatmaps include all 600 frames so the first two context frames are
visible; only frames 2–599 were included in the reported forecast metrics.

The notebook did not save its trained parameter tree or prediction arrays.
Therefore, the PNG here is the exact figure preserved in the executed notebook,
not a newly generated approximation. Reproducing additional residual or
per-neuron plots requires rerunning the fit and saving `heldout_prediction`.

## Temporal-shift diagnostic

The fit was reproduced to save its parameters and numerical predictions. The
reproduction reached training R² = 0.926793 and forecast R² = -0.487250, close
to the original notebook's training R² = 0.925482 and forecast R² = -0.451123.

A non-circular integer-lag scan tested whether forecast failure was merely one
temporal phase offset:

| Alignment | Unshifted R² on matched window | Shifted R² | Result |
|---|---:|---:|---|
| One lag shared by all trials | -0.481326 | -0.484532 | -7 frames; no improvement |
| Separate oracle lag per trial | -0.325358 | 0.127172 | substantial but incomplete improvement |

The independently selected trial lags were +250, -11, +42, +191, -179, and -60
frames. Their large variation shows that the visible displacement is not one
consistent clock offset. Initial-condition, direction, and/or speed errors vary
across held-out trajectories. The +250-frame result also reached the configured
search boundary, indicating that the first held-out trial is especially poorly
matched rather than being cleanly corrected by a small phase shift.

Oracle alignment uses the true future to select each lag. It is only a diagnosis
of error structure and must not be reported as forecasting performance.
