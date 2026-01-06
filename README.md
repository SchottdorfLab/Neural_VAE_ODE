# Neural VAE
* Written by: Kathleen Higgins
* Built for: Schottdorf Lab

## Most Recent Updates:
***January 6th, 6:12pm:***
- Added a v6 of the script. It removes PCA use. v5 still uses PCA.
- Updated both v5 and v6 to do an R2 sweep path across dimensions. 
- `v6_neural_vae.py`: "new R² sweep path that uses MATLAB’s random trial split (90/10), repeats per dimension, and plots black per‑trial dots + red overall dot for each latent dimension." 
- `v5_neural_vae.py`: "same sweep logic, but trains in PCA space and computes R² in raw neuron space via inverse PCA + de‑normalization (Option B)." 

- `config.txt`: added r2_sweep_* settings (disabled by default).
- Notes on choices made: 
    - Trial split matches MATLAB: random 10% test with seed 42. 
    - Multiple repeats per dimension = re-randomized splits (seed + repeat index).
    - v5 sweep uses training-only normalization and PCA, then evaluates R2 in raw neuron space ("option B")
***December 18th, 1:17am:***
- Added transition‑aware regularization with warmup (lambda_transition, lambda_transition_warmup_epochs) and optional trial‑level landmark sampling for the transition loss (transition_landmark_count).
- Transition loss now compares decoded dynamics (xhat[t+1]-xhat[t]) rather than re‑decoding predicted latents.
- Added a soft LLE constraint in latent space (lambda_lle, lle_k, lle_max_points, lle_temperature) to encourage local linearity without strict next‑step accuracy. This was the crucial piece of the most recent changes. 
- Fixed LLE distance computation to avoid in‑place ops that break autograd.
- Removed the graph‑based latent smoothness term (kNN graph loss) (decreased R^2 value). 

***November 25th, 10:55pm:***
- Increased # of experts to 8, now reaching a 0.4851 R^2 value. 

***November 25th, 4:58pm:***
- Reaching an R^2 value of 0.48 on real-world data. Added mutliple versions of decoders, the best decoder currently takes an MoE approach (MoeDecoder)
- Next step is to build v4 of the model, focusing on capturing neuron-level irregularities better and fully switching experts, with a decreased focus on smooth latent dynamics (smoothess 0.98 -> 0.86).
- The model is capturing the shape of neural activity better (R^2 is rising), but not capturing local variability/spikes well (MSE stays high). 
- Still indicating some averaging across neurons (looking @ the recon loss value)

***November 16th, 3:49pm:***
- Completed run of most recent data, reaching 0.29 R^2 on e65 data.
    -  Addressing stiff latent dynamic issues (smooth term increases from 0.97 -> 2.05)

***November 16th, 2:46pm:***
- Added scripts v2 (`v2_neural_vae.py`) and v3 (`v3_neural_vae.py`) of the scripts, renamed v1 from `neural_ode_vae.py` to `v1_neural_vae.py`. 
- v1 takes a global approach to the model (baseline structure for v2 and v3), v2 uses MoE plus changes to the underlying architecture in both the encoder and the decoder (unsuccessful), v3 is a minimally modified version of v1 that uses an MoE based encoder. 
- v3 ran a 0.78 R^2 on simulated data, testing ongoing on e65 data.

***November 10th, 9:15am:***
- Added sandbox and experiment 1 (Dr. Schottdorf) from the matlab code on E65. 

***November 8th, 2:12pm:***
- Scaled dimensions (to 5 dimensions) and increased noise to 2.0. Achieved a final R^2 of 0.9133, number of holdout trials is 3, final validation loss of 0.10623. Commit key is `0806572e9b8995251162795e461def6ad15fd882`. 

***October 31st, 1:01pm:***
- Updated storing of meta to get the correct data for the data visualization script. 
- Completed run of ```analyze_model.py``` with the 2D simulated data. 

***October 30th, 4:41pm:***
- Added logvar clamping to prevent extreme variance from inflating the reconstruction term.
- Added time normalization (tvec / tvec[-1]) to help the model behavor consistently across datasets. 
- Smaller step size for RK4 (slightly less aggressive trajectory fitting)
- Seed scan:
    - Ran the model on simulated data using 5 different seeds, saved the result dump to ```seed_sweep_results.txt```. 
    - Seed 1 is currently getting the best results with the simulated data (R^2 of 0.9789)

## File Structure
```
src/
├── helper_scripts/
├── mat_E65_data/
├── npz_e65_data/E65_data.npz
├── pt_files/
│   ├── ode_vae_best.pt
│   ├── final_metrics.pt
├── config.txt
├── preview.png
└── training_results.txt
```
### Data_Visualization: 
Holds ```analyze_model.py```, which runs on the data from the model to create visual analyses. Also contains .png images of the data visualizations. 

### Helper_Scripts:
Code primarily used to covert Matlab files (e.g. the E65 data) from a .mat file to a .npz file.

### Mat_E65_Data:
Holds the .mat files in various formats and versions from E65. 

### NPZ_E65_Data:
Holds the data converted from a .mat file to a .npz file.

### PT_Files: 
Contains the best model and final metrics.

## How to Run:
1. Configure the config.txt file. This file is used as the input configurations for the model.
2. Run the model. Ensure you are in the src directory, then type ```python3 neural_ode_vae.py```.
3. Results, in addition to being output into the terminal, will also be saved to training_results.txt. Additionally, an image of the training process will be saved to preview.png. 

