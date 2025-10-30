# Neural VAE
* Written by: Kathleen Higgins
* Built for: Schottdorf Lab

## Current Challenges:
Moving the data from the simulated data and scaling, also re-running data visualizations on the new model output. 
## Most Recent Updates:

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

