# Neural VAE
* Written by: Kathleen Higgins
* Built for: Schottdorf Lab

## Most Recent Updates:

November 25th, 4:58pm:
- Reaching an R^2 value of 0.48 on real-world data. Added mutliple versions of decoders, the best decoder currently takes an MoE approach (MoeDecoder)
- Next step is to build v4 of the model, focusing on capturing neuron-level irregularities better and fully switching experts, with a decreased focus on smooth latent dynamics (smoothess 0.98 -> 0.86).
- The model is capturing the shape of neural activity better (R^2 is rising), but not capturing local variability/spikes well (MSE stays high). 
- Still indicating some averaging across neurons (looking @ the recon loss value)

November 16th, 3:49pm:
- Completed run of most recent data, reaching 0.29 R^2 on e65 data.
    -  Addressing stiff latent dynamic issues (smooth term increases from 0.97 -> 2.05)
    
November 16th, 2:46pm:
- Added scripts v2 (`v2_neural_vae.py`) and v3 (`v3_neural_vae.py`) of the scripts, renamed v1 from `neural_ode_vae.py` to `v1_neural_vae.py`. 
- v1 takes a global approach to the model (baseline structure for v2 and v3), v2 uses MoE plus changes to the underlying architecture in both the encoder and the decoder (unsuccessful), v3 is a minimally modified version of v1 that uses an MoE based encoder. 
- v3 ran a 0.78 R^2 on simulated data, testing ongoing on e65 data.

November 10th, 9:15am:
- Added sandbox and experiment 1 (Dr. Schottdorf) from the matlab code on E65. 

November 8th, 2:12pm: 
- Scaled dimensions (to 5 dimensions) and increased noise to 2.0. Achieved a final R^2 of 0.9133, number of holdout trials is 3, final validation loss of 0.10623. Commit key is `0806572e9b8995251162795e461def6ad15fd882`. 

October 31st, 1:01pm:
- Updated storing of meta to get the correct data for the data visualization script. 
- Completed run of ```analyze_model.py``` with the 2D simulated data. 

October 30th, 4:41pm:
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

