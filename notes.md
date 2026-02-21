# Notes:
By Kathleen Higgins

## February 20th; 8:14pm:
- Added pearson r (but not running it tonight because if I get a bad score I will literally panic)

## February 20th; 4:52pm:

CONFIGURABLE COMMAND TEMPLATE:
```
python scripts/run_experiment.py --script src/v5_neural_vae.py --config configs/v5_base.txt --run-id 2026-02-20_v5_pca --note "baseline v5"
```

- I got really fed up with constantly losing track of the work that I did before, and not knowing what changes I had made and what scores they gave me.
- Considering this, I established some code to help me out. To run it, you type the following (this is just an example on v5)

```
python scripts/run_experiment.py --script src/v5_neural_vae.py --config configs/v5_base.txt --note "baseline v5"
```
- What this does is it creates runs/<runid> with: 
    - run.json wwhich contains metadata, code summary, and metrics. 
    - run.log
    - config.txt 
    - config_original.txt
    - copied artifacts ex. best checkpoint, preview, metrics 
    - Also appends a row to index.csv. 
- I also wanted to add an AI/ML summary of code changes that reads stdin and returns a one-sentance summary. 
- To do this, you can set LLM_SUMMARY_CMD to any CLI that reads from stdin: 

```
$env:LLM_SUMMARY_CMD = "ollama run llama3"
python scripts/run_experiment.py --script src/v5_neural_vae.py --config configs/v5_base.txt
```
- What this script does: it pipes a prompt and a git diff into that command and stores the first line under run.json under code_change_summary. If no LLM is set, it falls back to a deterministic one-line summary based on git diff --stat. 
## February 20th, 4:49pm:
Thoughts:
- Also compute r, not just R2, so we can have a one-to-one comparison with Dr. Schottdorf's code in the MIND algorithm. 

## January 10th, 1:45pm:
https://www.biorxiv.org/content/10.1101/418939v2.full#F5
```
Paper techniques that can actually move R² in your ODE‑VAE
These are the most “transferable” MIND ideas for improving reconstruction quality (not just adding bells/whistles):

Match the paper’s calcium preprocessing (if E65 is calcium, it is ~15 Hz)
They smooth with an 11‑bin Gaussian and threshold at 4σ (robust σ per neuron).
For recon traces, they do baseline subtraction before smoothing/threshold.
Why it helps R²: it de‑emphasizes tiny amplitude noise and emphasizes structured transients that are more reproducible trial‑to‑trial.
Use the paper’s exact cross‑validation structure
Hold out random trials, repeat ~10 times, average R² (and plot per‑trial dots).
Your training currently uses small holdout counts sometimes; high variance in val can hide real improvements and encourages overfitting.
Stop throwing away data points (MIND fits manifolds to all points)
In MIND they fit on essentially all eligible timepoints; landmarks are for efficiency/graph construction, not to shrink the dataset.
In your ODE‑VAE, subsampling trials/timepoints can reduce generalization and hurt R².
Add an explicitly local “mapping” component (MIND uses LLE / local regression)
Your decoders are global (MLP/MoE/neuron-aware). MIND’s advantage often comes from local reconstruction in manifold space.
A strong hybrid for R²: decoder = global MLP + a local neighbor-based residual (kNN/LLE-like over a small set of prototype latent states). This directly targets the “local irregularities/spikes” problem without forcing the ODE to memorize them.
Use robust per-neuron scaling/weighting (robust σ)
The paper normalizes/thresholds per neuron using robust σ. In an MSE objective, high-variance neurons dominate unless you normalize correctly.
A simple improvement aimed at R²: compute recon loss in “variance-normalized units” (roughly SSE/var), which aligns training more closely with maximizing R².
Align your time discretization with the data
E65 is ~15 Hz (Δt≈67 ms). If you resample heavily or use a mismatched fps, you can blur transitions and lose R².
Using the native frame grid (or resampling to a consistent but realistic L) usually improves both dynamics fit and reconstruction.
```

## January 10th, 11:31am:
"Based on the diagram and the pasted paper text, answer two questions: 1. What metrics should I be focusing on to answer similar questions to those addressed in the text? 2. What techniques could I add from this paper that would improve metrics?"

1. Metrics to focus on (to answer the same questions as the paper):
### Cross‑validated decoding index (Pearson r) on vectorized neuron×time blocks
- For “population reconstruction” (panels a–b): correlate vec(X_test) vs vec(X̂_test) and average across repeated held‑out trials/splits.
- This is the paper’s primary “decoding index” metric.
### Variance‑explained R² on the same blocks (optional but useful)
- `1 − var(X−X̂)/var(X)`; MATLAB uses this for the “Crossval R²” plot you showed.
- Track both r and R² because r is scale‑invariant while R² penalizes bias/scale errors.
### Saturation curves vs embedding dimension d
- Plot decoding index (and/or R²) as a function of latent_dim and look for where it plateaus (their “minimum dimensions needed” idea).
### Intrinsic dimension estimate from distance scaling (if you want the “correlation dimension” analysis)
- Build a geodesic distance matrix ρ(i,j) (in MIND it’s from transition probabilities) and fit the slope of log N(r) vs log r over a scaling range to estimate intrinsic d.
### Single‑neuron predictability metric (panel c/e)
- For held‑out neuron i: correlation between true and predicted neuron activity across time (and trials), averaged over 5 folds and multiple neurons.
## 2. Paper techniques worth adding (to improve those metrics)
### Match the evaluation preprocessing used in the paper (biggest “apples‑to‑apples” improvement)
- Apply the same calcium-style transform before scoring: 11‑bin Gaussian smoothing + 4σ threshold per neuron using robust σ, and baseline subtraction for recon traces. This often increases correlation‑based decoding index because it emphasizes transient timing over small amplitude errors.
### Trial‑level cross‑validation protocol
- Use repeated random held‑out trials (10 repeats like the paper) and report mean±SEM. Your current setup can be made directly comparable by scoring r on vectorized blocks.
### Held‑out neuron decoding evaluation
- Add the panel‑c style test: fit latents on N−1 neurons, then learn g(latents)→neuron_i. In the paper they use Gaussian Process Regression with an RBF kernel; in Python you could start with a small smooth regressor (or a GP if tractable) and measure cross‑validated r.
### Noise/thresholded observation model
- MIND explicitly accounts for calcium transient nonlinearity via thresholding. For your VAE, consider an observation model that’s closer to calcium data (e.g., zero‑inflated / hurdle / rectified Gaussian, or simply threshold‑aware loss). This can improve correlation‑based reconstruction metrics.
### Intrinsic-dimension analysis via geodesic distances
- If you want the “d≈4–6” type claim, add a geodesic-distance–based intrinsic dimension estimator (kNN graph shortest paths in latent space is a practical proxy), then do the N(r) ~ r^d slope fit and bootstrap CIs.

## January 7th, 8:02pm:
- Realized all my commits from the Schottdorf lab computer are coming from Mubariz. That's not good. Setting up stuff to work with my Github, and needed a specific key. "Sunshine". Here is some output from the terminal print statement. 
```
(base) PS C:\Users\schot\Neural_VAE_ODE> ssh-keygen -t ed25519 -C "kathigg@udel.edu"
Generating public/private ed25519 key pair.
Enter file in which to save the key (C:\Users\schot/.ssh/id_ed25519): key.txt
Your identification has been saved in key.txt
SHA256:sS82S+X1J60n7GJ2HdLZa/lllqZNtsX0A6r2Vtj3Ar0 kathigg@udel.edu
The key's randomart image is:
+--[ED25519 256]--+
|                 |
|                 |
|        .        |
|         o       |
|        S . +o. +|
|         + oo=oO+|
|        = o..o=O#|
|       o =..+ EXO|
|        o.o+ =+*o|
+----[SHA256]-----+

(base) PS C:\Users\schot\Neural_VAE_ODE>
```

## January 7th, 2:38pm:
- Quick notes: first run of v6 was absolutely unreal, in terms of how horrible it was (6.... reconstruction error, relatively low trial error). Not sure why?
- With v5, I'm seeing an R2 value of 0.45. I believe that's a relative drop compared to previous work at 0.5, and if I am remembering correctly, I also previously saw a much lower reconstruction error. So that lower score is as R2 is being computed in raw neuron space, via inverse PCA and de-normalization.
- Decided to keep evaluation of R2 in raw neuron space, not in PCA space---I'm training in PCA space, and then evaluating R2 in the raw neuron space in v5, which is the more "true" metric for reconstruction quality, because it reflects the original neuron scale and variability, not fidelity ot the compressed (PCA) representation. 

## January 7th, 2:30pm:
- Added print statement to the top of scripts v1 through v6 to (theoretically) copy all outputs in the terminal while running the script to a designated output file. 
- This output file is overwritten for each new run of any script v1 through v6. 
## January 7th, 11:20am:
- Do you remember how the lab computer just randomly decided to disconnect from DNS? Well, it's connected now when I showed up this morning, fully working with DNS, same IP address, and it completely closed out my terminal windows---it may have restarted and flushed its DNS cache. 
- Ran v6 of the script. 
- Updated of v6 from v5:
    - No PCA
    - Changed v6 to hold out last K trials, if later trials differ (drift, task, changes), val loss spikes while train keeps improving. 
    - Landmark subsampling bias. Greedy coverage on flattened sequences can over-represent "exteme" trials; the model then underfits the average val trials. 
    - Just general issues with regularizers dominating generalization (same stuff as before)
    - Normalization mismatch: v6 uses make_sequences (z-scores per neuron after overriding ROI, if the val trials have different per-neuron stats, z-score is leaking into train stats into val and hurting reconstruction.)
**Fixes**: 
- Re-enable PCA
- Change split to random trial split or whatever I had before
- Change back to the previous version fo landmark subsampling (this is my hunch)

## January 6th, 6:12pm:
- Added a v6 of the script. It removes PCA use. v5 still uses PCA.
- Updated both v5 and v6 to do an R2 sweep path across dimensions. 
- `v6_neural_vae.py`: "new R² sweep path that uses MATLAB’s random trial split (90/10), repeats per dimension, and plots black per‑trial dots + red overall dot for each latent dimension." 
- `v5_neural_vae.py`: "same sweep logic, but trains in PCA space and computes R² in raw neuron space via inverse PCA + de‑normalization (Option B)." 

- `config.txt`: added r2_sweep_* settings (disabled by default).
- Notes on choices made: 
    - Trial split matches MATLAB: random 10% test with seed 42. 
    - Multiple repeats per dimension = re-randomized splits (seed + repeat index).
    - v5 sweep uses training-only normalization and PCA, then evaluates R2 in raw neuron space ("option B")

## January 6th, 2026, 5:49pm:
Quick reminder, to login to the Schottdorf lab computer, it's:
```
ssh schot@128.175.181.203
```
And I authenticate using a key and a password. 

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
