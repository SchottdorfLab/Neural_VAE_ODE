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
