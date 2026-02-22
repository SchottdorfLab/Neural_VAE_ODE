# Neural VAE
* Written by: Kathleen Higgins
* Built for: Schottdorf Lab

## February 22nd, 2:34pm:
I was really thinking about lambda smooth and whether adding that constraint is supressing legitimate neural variability, and whether it's fighting against the transition loss (matching dx/dt), but like it's already heavily reduced in terms of smoothness. 
Ran it with 0, and stuff is a good bit more jittery, but it's kind of providng that the argument that it's redundant with the ODE isn't strictly correct. 

The ODE gives differentiability, the smoothness term specifically controls trajectory speed. 

With lambda_smooth = 0.0:
```
Final r: 0.6853
Final R²: 0.4645
```
So slight drop. Not major. But I'm going to bring it back to what it was before. So the prediciton that lambda_smooth is hurting variance scores: disproven. 

It's my own hypothesis, so I'm not insulted, but it's an interesting little nugget of data. What if we increased the lambda_smooth? Is it time to start fine-tuning hyperparameters by doing a hyperparameter sweep?

## February 22nd, 1:35pm:
- Panic resolved. 
```
Final r: 0.6920
Final R²: 0.4731
```
Changes made to avoid data leakage: 


Still potential issues: landmark selection is happening before the split, meaning (theoretically) val trials can influence which trials are kept, but I'm so tired and the other data leakage things I worried about (and what I changed) didn't impact scores pratically at all, after the crazy cocktail of changes I did where I ended up with a ~-0.2 R2. Btw, the changes I made that did all that were:
1. Trial-split MIND Style:
```
MIND uses a random 90/10 split of trials:
rand(length(trials_all),1) > 0.1
I added the same in v5:
mind_split_enabled = true
mind_test_frac = 0.1
mind_split_seed = 42
So we’re no longer using holdout_trials or “last K trials.” It’s a random 10% test set like the MIND script.
```
2. Training-only normalization: honestly, I think this is what blew everything up. I need z-scoring, bro. In the OG v5 (before stuff blew up), I globally z-scored all trials together inside make_sequences. 
```
In MIND mode, I do it like the MATLAB script:

Compute mu and sd only on the training trials.
Normalize train and test using those training stats.
Why this matters:

It prevents test data from “leaking” into preprocessing.
This matches proper cross‑validation: train preprocessing only.
```
So this is now reverted, but with a caveat---we fixed all of the leakage associated with training by moving when PCA and z-scoring happen so the model doesn't get sneak peeks (that's a poor way to say that, but you know what I mean).

3. Training only PCA. 
```
In the original v5 flow, PCA is fit on all trials (because PCA happens before splitting).

In MIND mode:

Fit PCA only on training trials.
Apply that PCA model to the test trials.
That’s exactly what the MATLAB pipeline is doing (fit PCA on training, transform test, then invert to raw for reconstruction).
```

4. I also think this part screwed me over to, hence why the R2 was horrific----but I think this was much less impactful than the z-scoring thing. To evalute this hypothesis, Ima check this later and run R2 in the same way as mind styles, computing a global R2 on the full flattened test set like MIND. Rn, I have averaged per-batch R2. Which I feel like is fine. But we'll see what Dr. Schottdorf says after he reads my paper. Mayhap it would be nice to have a one-to-one comparison between ways to evaluate R2. 

**NOTE TO SELF: GO BACK AND RUN WITH THE UPDATED R2 METRICS THAT MIND USES.**

The MIND uses:
```
R² = 1 - var(raw_data(:) - reconstructed_data(:)) / var(raw_data(:))
```

## February 22nd, 1:24pm:
Thoughts on leaky behavior---preprocessing in the old pipeline.
1. Loading all trials.
2. I z-scored and PCA-reduced the full dataset. 
3. Then I split into train/val. 
This means (theoretically) that test trials influenced the mean/std and PCA basis used for training. Thought: Even if the model didn't train on the test trials, the preprocessing saw them. 
Where'd it happen? in make_sequences() where we z-score on the full roi. 
Plus, PCA was fit on full roi before splitting into train/val. 
Possibly the leakage was global normalization + global PCa before the split. 
Action: Goig to try to change the preprocessing order to fit z-score plus PCA on train only, then apply to val. 

## February 22nd, 12:23pm:
- Changed to be more like MIND in how we're computing R2 (hopefully this doesn't blow stuff up). We also changed to do 90/10 split like MIND. No idea what the results are going to be, so wish me luck. 
- Hey so it's a random 10% of trials that are tested, same as Matlab. You can enable it by going into config.txt and changing mind_test_frac = 0.1. 
- Terrible. Negative R2. Not sure why, but it's not just a question of how we're running stuff. 

## February 22nd, 12:08pm:
- So I made the GRU encoder but didn't have a chance to test it. It is super volatile. 
```
Final r: 0.5579
Final R²: 0.2505
```
Generally, thinking about how the MIND algorithm did data eval, and how to make it more similar so we can cover data in the same way and do a clean one-to-one comparison. 

## February 21st, 4:45pm:
- IMPORTANT COMMAND ALERT. I got tired of constantly having to go back and copy-paste my latest config file, so I added this:
```
python scripts/restore_last_config.py --dest configs/v5_base.txt
```
And also, you can add flags to decide how far back you want to go with the config file, e.g.:
```
python scripts/restore_last_config.py --dest configs/v5_base.txt --offset 2
```
Offset 1 is the latest, offset 2 is previous. 


## February 21st, 4:37pm:
- Note to self: running bidirectional with a transformer makes it worse (weird??)
```
Final r: 0.6739
Final R²: 0.4421
```

## February 21st, 4:04pm:
- Latest run with modifications to lr, lambda_transition, lambda_lle, etc. made scores go slightly down---not sure why. I'm tired of hyperparameter tuning and I'd like to get larger stuff to work. So i'm going to try the other versions. 

## February 21st, 3:39pm:
- With an updated dimensionality of 7 and a new pool of first:
```
Final r: 0.6969
Final R²: 0.4834
```
So better again. 
## February 21st, 3:21pm: 
- Using a transformer model, better with scores (yay!)
- Previous run had been a little too heavy for the transformer model with a dropout of 0.1 and two layers, I switched it to a layer of 1 and no dropout and we're rolling vastly better, to the extent that we're better than non-transformer "first" version. 

New metrics:
```
Final r: 0.6915
Final R²: 0.4736

```

## February 21st, 3:16pm:
```
What the latest transformer run shows
From run_metadata.json:

r2 = 0.3994 (lower than baseline ~0.4506)
r = 0.6325
recon = 2.0688 (worse)
kl = 1.9649 (lower)
smooth = 1.7922 (lower)
Compared to the non‑transformer run (run_metadata.json), recon got worse and KL dropped a lot. Lower KL typically means the posterior is closer to the prior → less information in z0, which hurts reconstruction.

Why it’s likely lower

Mean pooling + transformer can wash out dynamics
encoder_pool = mean produces a time‑average representation, but z0 is an initial condition. You’re summarizing the entire sequence into one vector, which can blur the info needed to reconstruct early dynamics.

Transformer is data‑hungry
You only have ~100 sequences. A 2‑layer transformer with dropout can underfit or overly smooth in this regime.

Strong regularizers stay on
Smoothness/transition/LLE penalties are still active. With a more expressive encoder, these can over‑constrain and push toward overly smooth latents, reducing recon and R².

Concrete fixes to try first

Change pooling to anchor t0 but still use full sequence
Set encoder_pool = first. This uses attention over all frames but takes the first position embedding as the summary (better aligned with z0).

Shrink the transformer
Start small:

encoder_layers = 1
encoder_heads = 2
encoder_hidden = 128
encoder_ffn_dim = 256
encoder_dropout = 0.0
Reduce regularizers for the first transformer sweep
Try:

lambda_transition = 0.0
lambda_lle = 0.0
lambda_smooth = 0.0002
Then re‑introduce once the encoder shows a gain.

Lower LR
Transformers often like smaller LR:

lr = 0.001
Check best‑epoch vs final‑epoch
The run metadata uses the last epoch’s mean_r2, not necessarily the best. If R² peaked earlier, you’re comparing the wrong point.
```

## February 21st, 3:06pm:
- Just ran using the transformer 
## February 21st, 3:01pm:
Quick note of the previous first config values:
```
encoder_type = first
encoder_hidden = 256
encoder_layers = 1
encoder_bidirectional = false
encoder_dropout = 0.0
encoder_heads = 4
encoder_ffn_dim = 512
encoder_pool = mean
```

Also a reminder that the new config information is stored in v5_base.txt, and not in config.txt anymore. Config.txt can still be used if you're running directly on the script, but if you're running using the experiments, that's not going to work.

## February 21st, 1:42pm: 
**What's mu?**
Mu is the mean of the encoder's approximate posterior over the initial latent state---i.e. q(z0 | x) if using a first-frame system, or if we're using a sequence encoder, it's q(z0 | z1:L). 
- The encoder outputs mu and logvar.
- The model samples z0 via the reparameterizatino trick (described more below). eps comes from the normal distribution N(0, I). 
- mu therefore represents the central, e.g. most likely latent initial condition inferred from the input sequence. 

Some more quick notes on that:
- If you set eps = 0, then z0 = mu is the deterministic latent initial state.
- logvar controls uncertainty around that mean, a larger logvar means more stochasticity. 
***To summarize: mu is the inferred latent "starting point" for the ODE trajectory.***

**What's z0?**
- z0 is the initial latent state for a trial. 
- It is the latent starting point that the ODE evolves over time. 
- We sample z0 with the reparameterization trick: z0 = mu + exp(0.5 * logvar) * eps
- That z0 is then fed into the latent ODE, which produces the full latent trajectory z(t) over the trial. 
- The decoder maps z(t) back to predicted neural activity x̂(t). 

## February 21st, 1:39pm:
Quick explanation of the different encoder types implemented in v5 (previously, we'd been using one that only uses the first frame, e.g. encoder_type = first).

Now, we've introduced three new encoder types, which all avoid the "first-frame only" issue. Any of these will allow the model to condition q(z0) on the full sequence x1:L. 

**GRU**
- encoder_type = gru
- It runs a recurrent model over the full sequence and uses the finla hidden state to produce mu, logvar. 
- It is a good default for modest data sizes, captures temporal order and dynamics, plus it's stable and cheap (slay).
- Weaknesses is that it can underfit long-range structure compared to attention, less parallelizable. 

**Transformer**
- encoder_type = transformer
- It projects each frame, adds positional encoding, runs a Transformer encoder, then pools over time (mean/first/last) to get mu, logvar.
- Strengths: strong long-range modeling, highly parallel, flexible pooling.
- Weaknesses: more parameters and compute plus it often needs more data or regularization to beat GRU.

**Temporal Attention**
- encoder_type = attn
- It does: MLP per frame + learned attention weights over time, then a weighted su to get a single vector for mu, logvar.
- Stengths: simplest "full sequence" model + it's cheap and it has interpretable attention weights.
- Weaknesses: it doesn't model interactions between timepoints as richly as a transformer. 


## February 21st, 1:29pm:
- Final r: 0.6734
- Final R²: 0.4506
***Note: first time I've been able to see pearson's r value***. 

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
- Realized all my commits from the Schottdorf lab computer are coming from Mubariz. That's not good. Setting up stuff to work with my Github, and needed a specific key. Here is some output from the terminal print statement. 
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

## December 18th, 1:17am:
- Added transition‑aware regularization with warmup (lambda_transition, lambda_transition_warmup_epochs) and optional trial‑level landmark sampling for the transition loss (transition_landmark_count).
- Transition loss now compares decoded dynamics (xhat[t+1]-xhat[t]) rather than re‑decoding predicted latents.
- Added a soft LLE constraint in latent space (lambda_lle, lle_k, lle_max_points, lle_temperature) to encourage local linearity without strict next‑step accuracy. This was the crucial piece of the most recent changes. 
- Fixed LLE distance computation to avoid in‑place ops that break autograd.
- Removed the graph‑based latent smoothness term (kNN graph loss) (decreased R^2 value). 

## November 25th, 10:55pm:
- Increased # of experts to 8, now reaching a 0.4851 R^2 value. 

## November 25th, 4:58pm:
- Reaching an R^2 value of 0.48 on real-world data. Added mutliple versions of decoders, the best decoder currently takes an MoE approach (MoeDecoder)
- Next step is to build v4 of the model, focusing on capturing neuron-level irregularities better and fully switching experts, with a decreased focus on smooth latent dynamics (smoothess 0.98 -> 0.86).
- The model is capturing the shape of neural activity better (R^2 is rising), but not capturing local variability/spikes well (MSE stays high). 
- Still indicating some averaging across neurons (looking @ the recon loss value)

## November 16th, 3:49pm:
- Completed run of most recent data, reaching 0.29 R^2 on e65 data.
    -  Addressing stiff latent dynamic issues (smooth term increases from 0.97 -> 2.05)

## November 16th, 2:46pm:
- Added scripts v2 (`v2_neural_vae.py`) and v3 (`v3_neural_vae.py`) of the scripts, renamed v1 from `neural_ode_vae.py` to `v1_neural_vae.py`. 
- v1 takes a global approach to the model (baseline structure for v2 and v3), v2 uses MoE plus changes to the underlying architecture in both the encoder and the decoder (unsuccessful), v3 is a minimally modified version of v1 that uses an MoE based encoder. 
- v3 ran a 0.78 R^2 on simulated data, testing ongoing on e65 data.

## November 10th, 9:15am:
- Added sandbox and experiment 1 (Dr. Schottdorf) from the matlab code on E65. 

## November 8th, 2:12pm:
- Scaled dimensions (to 5 dimensions) and increased noise to 2.0. Achieved a final R^2 of 0.9133, number of holdout trials is 3, final validation loss of 0.10623. Commit key is `0806572e9b8995251162795e461def6ad15fd882`. 

## October 31st, 1:01pm:
- Updated storing of meta to get the correct data for the data visualization script. 
- Completed run of ```analyze_model.py``` with the 2D simulated data. 

## October 30th, 4:41pm:***
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

