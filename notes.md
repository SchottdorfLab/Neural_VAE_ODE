# Notes:
By Kathleen Higgins


## January 6th, 2026, 5:23pm
Okay, so there are a couple problems.
1. Overfitting. R2 (according to a run by Codex on CPU) peaks at 0.48 around epoch 141 then drops by epoch 150, ending with a final R2 of 0.4368. 
2. A single smooth latent ODE (even an MoE) struggles to be both globally predictive and neuron-level accurate. 
3. PCA to 95% variance removes low-variance but behaviorally important spiky components---perhaps why the model performs worse on neurons with a big spike?
    - **Solution**: Try a run without PCA?
4. Another issue: training on a tiny datset. After dropping trials, I have ~178 trials avalible, but landmark_count reduces to 100 sequences, and then I only validated on holdout_trials=3. Which makes the validation metric high-variance and encourages overfitting late in training. 
    - **Solution**: Ask for more data to train on. 
5. Landmark selection "isn't trial level." I flatten X over time, pick landmarks among timepoints, then map back to trial indicies via mod. This could (according to Codex's analysis) duplication trials and bias which trials the VAE keeps (not how MIND uses landmarks).
    - **Solution**: Experiment with an alternate way to do landmark selection that is trial level? 
    - What even is landmark selection?
6. Another thought about eradicating PCA. MIND always reconstructs back to the original space via inverse PCA + learned mapping. In v5, the model learns in PCA component space (N=129) and "never explicitly optimizes reconstruction back to the 375-neuron space."
    - **Solution**: Get rid of PCA.
7. Checkpointing is broken (the best epoch isn't kept, ***is this even the optimal way to do this anyway? How is it normally done in studies; is it the best R2?***) 
    - **Solution**: Ask Codex to fix best_val save condition. 
8. "Transition + LLE losses are currently too weak to help, but still add gradient noise
In the run, trans ~0.03 and lle ~1e-4–1e-3. With lambda_transition=0.01 and lambda_lle=0.01, their contributions to the total loss are tiny vs recon/KL, so they won’t reshape the solution much—yet they introduce extra stochasticity (LLE subsampling) that can slightly degrade recon." 
    - **Solution**: No idea. 

**Solutions, according to Codex (not mine, my solutions are written above):**
```
"High-value ways to improve (most MIND-aligned)

Train on all trials; use landmarks only for regularization/visualization. Set landmark_count=0 (or much larger), and increase holdout_trials (e.g. 20) or do a random trial split. This alone usually improves stability and generalization.
Reconstruct in neuron space (MIND-style “inverse PCA” idea). Keep PCA for efficiency, but decode back through a fixed inverse PCA (or an explicit linear layer initialized from PCA components) and compute recon loss on the original 375D ROI. This is the most direct fix for “captures shape but misses spikes”.
Make dynamics less strictly deterministic. MIND’s forest is probabilistic local modeling; your latent ODE is deterministic. If you want MIND-like robustness, consider adding latent process noise (Neural SDE / stochastic latent ODE) or a learned per-step residual.
Model observation noise (PPCA-like). Replace plain MSE with Gaussian NLL with learned per-neuron (or per-dimension) variance; this mirrors PPCA’s explicit noise model and helps with heteroscedastic neurons.
Fix best-model saving + early stopping. Your run peaked higher than it ended; saving the best epoch will give you better practical performance immediately."
```

### January 6th, 2026, 4:50pm
To run Dr. Schottdorf's Matlab code in the Visual Studio Code terminal: 
```
conda deactivate
/Applications/MATLAB_R2025b.app/bin/matlab -nodesktop -nosplash
```

```
matlab
```

```
cd('/Users/kathleenhiggins/Neural_VAE_ODE');

addpath(genpath('sandbox/MIND_experiments/mind_core'));
addpath('sandbox/MIND_experiments');

cd('src/mat_E65_data');   % so exp1’s load('./E65.mat') works
exp1_embeddingCrossval
```
