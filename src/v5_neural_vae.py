"""
v5_neural_vae.py — MoE Latent Neural ODE VAE + transition regularization + soft LLE (CUDA-first)

What it does today:
- Data: loads E65 .npz from PATHS["data"] (note: config.txt also exists but the script is still largely path-driven).
- Preprocessing: PCA to retain ~95% variance (hard-coded), then trial grouping + resampling to fixed length.
- Normalizes per neuron (session z-score) and time-normalizes tvec to [0,1].
- Optional landmark subsampling via greedy coverage on flattened sequences (speed optimization).
- Optional per-trial baseline correction (subtract mean of first 5 frames).

Model architecture:
- Encoder: MLP x(t0) -> (mu, logvar).
- Latent dynamics: MoELatentODEFunc (mixture of expert ODE vector fields).
- Decoder options (decoder_type): MLP / NeuronAware / LocalAttention / MoE decoder.

Loss / regularizers (in addition to recon + KL + smoothness):
- Transition-aware regularization (lambda_transition):
  penalizes mismatch of decoded dynamics Δx̂(t)=x̂(t+1)-x̂(t) vs Δx(t) (with linear warmup).
  Optionally computed on only a subset of trials per batch (transition_landmark_count) to avoid over-regularizing noisy trials.
- Soft LLE latent constraint (lambda_lle):
  flattens latent trajectory points, finds kNN in latent space, and reconstructs each point from neighbors
  using softmax weights; penalizes reconstruction error to encourage locally linear structure.

Compute/compatibility:
- Device selection is CUDA-first, then CPU (no MPS path).
- Uses torchdiffeq odeint (dopri5) for latent integration.

Outputs:
- Saves metrics/plots/checkpoints under pt_files/ and logs training to training_results.txt.
"""

# Written by Kathleen Higgins
# Worked as of 2025-09-10
# src/v5_neural_vae.py
# most recent version of the neural ODE VAE for neural data

import os, math, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
from torchdiffeq import odeint
import datetime
import random
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import json
import hashlib
import joblib

# Utility: convert NumPy/tensor types into JSON-friendly Python types
def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj

# used to compute file sha256 checksum, used to log the hash of the input file (for reproducibility)
def compute_file_sha(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()

# used to set the seed, later used to sweep across seeds 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # allow non-deterministic ops for exploration

    # ensures CuDNN kernels behave deterministically 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#____________________metrics__________________#
def compute_r2(y_true, y_pred):
    """
    Compute coefficient of determination (R^2) between ground truth and prediction.
    Works with torch tensors or numpy arrays.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan  # Avoid divide-by-zero for constant sequences
    return 1 - (ss_res / ss_tot)

# --- Root directory auto-detection --- #
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# --- Root directory auto-detection --- #
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data":          os.path.join(SRC_DIR, "npz_e65_data", "E65_data.npz"),
    # "data":          os.path.join(SRC_DIR, "random_walk_data", "synthetic_rat_data.npz"),
    "out_dir":       os.path.join(SRC_DIR, "pt_files"),
    "final_metrics": os.path.join(SRC_DIR, "pt_files", "final_metrics.pt"),
    "preview":       os.path.join(SRC_DIR, "preview.png"),
    "training_log":  os.path.join(SRC_DIR, "training_results.txt"),
    "config":        os.path.join(SRC_DIR, "config.txt"),
}
os.makedirs(PATHS["out_dir"], exist_ok=True)


def load_config_from_txt(path):
    """Load key=value pairs from a text file into a dict."""
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = [x.strip() for x in line.split("=", 1)]
                # try casting to float or int when possible
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                elif "." in v and v.replace(".", "", 1).isdigit():
                    v = float(v)
                elif v.isdigit():
                    v = int(v)
                cfg[k] = v
    return cfg

#____________________utils__________________#
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def zscore_per_neuron(x):
    # x: [T, N]
    mu = x.mean(axis = 0, keepdims=True)
    sd = x.std(axis = 0, keepdims=True) + 1e-8
    return (x- mu) / sd, mu, sd

def group_indices_by_trial(trial_vec):
    # trial_vec: (5,)
    # returns list of index arrays, one per trial in ascending order
    trial_vec = np.asarray(trial_vec).astype(int)
    trials = np.unique(trial_vec)
    groups = []
    for tr in trials:
        idx = np.where(trial_vec == tr)[0]
        if idx.size > 0:
            groups.append(idx)
    return trials, groups

def resample_sequence(x, t_src, L, t0 = None, t1 = None): 
    """
    x: [Ts, N] values at times t_src[Ts]
    return x_rs: [L, N] resampled on L points between t0..t1
    """

    Ts, N = x.shape
    if t0 is None: t0 = float(t_src[0])
    if t1 is None: t1 = float(t_src[-1]) if t_src[-1] > t_src[0] else (t_src[0]+ 1.0)
    t_dst = np.linspace(t0, t1, L, dtype=np.float32)
    # vectorized 1D linear interpolation for each neuron
    # fallback to numpy.interp per neuron to keep code simple and robust
    x_rs = np.empty((L, N), dtype=np.float32)
    t_src_np = np.asarray(t_src, dtype=np.float32)
    for j in range(N):
        x_rs[:, j] = np.interp(t_dst, t_src_np, x[:, j])
    return x_rs, t_dst

def make_sequences(npz, trial_len_s=12.0, fps=10.0, drop_first_trials=10, min_frames=10):
    """
    Build fixed-length oer-trial sequences by:
    - grouping frames by trial
    - resampling each trial to L frames (L = trial_len_s * fps)
    Returns: 
    X: [B, L, N], tvec: [L], meta: dict
    """

    roi = npz["roi"] # [N, T] or [T, N] depending on the export 
    roi = (roi - roi.mean(axis=0)) / (roi.std(axis=0) + 1e-8)
    # thr printout showed roi shape (375, 7434) => (N, %). Transpose to [T, N]. 
    if roi.shape[0] < roi.shape[1]:
        # assume (N, T) -> (T, N)
        roi = roi.T
    T, N = roi.shape

    trial = npz["Trial"].astype(int) # (T, )
    time = npz["Time"].astype(float) # (T, )

    # zscore per neuron (session level)
    roi, mu, sd = zscore_per_neuron(roi)
    trials, groups = group_indices_by_trial(trial)

    # drop first K trials
    if drop_first_trials > 0 and len(trials) > drop_first_trials:
        keep_mask = trials >= trials[0] + drop_first_trials
        groups = [g for g, keep in zip(groups, keep_mask) if keep]
        trials = trials[keep_mask]

    L = int(round(trial_len_s * fps))
    X = []
    good_trial_ids = []
    for tr, idx in zip(trials, groups):
        if idx.size < min_frames: # skip trivial trials 
            continue
        x_tr = roi[idx, :] # [Ts, N]
        t_tr = time[idx] - time[idx][0] # start each trial at t=0
        x_rs, t_rs = resample_sequence(x_tr, t_tr, L, t0=0.0, t1=trial_len_s)
        X.append(x_rs)
        good_trial_ids.append(tr)

    if len(X) == 0:
        raise ValueError("No trials produced sequences (NOOOOO!!) Check trial/time vectors and args.")
    X = np.stack(X, axis=0).astype(np.float32) # [B, L, N]
    return X, t_rs.astype(np.float32), {"trials_used": np.array(good_trial_ids), "mu": mu, "sd": sd, "N": N}

def make_sequences_raw(npz, trial_len_s=12.0, fps=10.0, drop_first_trials=10, min_frames=10):
    """
    Build fixed-length per-trial sequences WITHOUT z-scoring.
    Returns:
    X_raw: [B, L, N], tvec: [L], trial_ids: [B]
    """
    roi = npz["roi"]
    if roi.shape[0] < roi.shape[1]:
        roi = roi.T
    trial = npz["Trial"].astype(int)
    time = npz["Time"].astype(float)

    trials, groups = group_indices_by_trial(trial)
    if drop_first_trials > 0 and len(trials) > drop_first_trials:
        keep_mask = trials >= trials[0] + drop_first_trials
        groups = [g for g, keep in zip(groups, keep_mask) if keep]
        trials = trials[keep_mask]

    L = int(round(trial_len_s * fps))
    X_raw = []
    trial_ids = []
    for tr, idx in zip(trials, groups):
        if idx.size < min_frames:
            continue
        x_tr = roi[idx, :]
        t_tr = time[idx] - time[idx][0]
        x_rs, t_rs = resample_sequence(x_tr, t_tr, L, t0=0.0, t1=trial_len_s)
        X_raw.append(x_rs)
        trial_ids.append(tr)

    if len(X_raw) == 0:
        raise ValueError("No trials produced sequences. Check trial/time vectors and args.")
    X_raw = np.stack(X_raw, axis=0).astype(np.float32)
    return X_raw, t_rs.astype(np.float32), np.array(trial_ids)

def normalize_sequences(X_raw, mu=None, sd=None):
    flat = X_raw.reshape(-1, X_raw.shape[-1])
    if mu is None:
        mu = flat.mean(axis=0, keepdims=True)
    if sd is None:
        sd = flat.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_raw - mu) / sd
    return X_norm, mu, sd

def baseline_correct_np(X):
    baseline = X[:, :5, :].mean(axis=1, keepdims=True)
    return X - baseline

def baseline_correct_torch(X):
    baseline = X[:, :5, :].mean(dim=1, keepdim=True)
    return X - baseline

def r2_var_explained(x, xhat):
    x = np.asarray(x)
    xhat = np.asarray(xhat)
    denom = np.var(x)
    if denom == 0:
        return np.nan
    return 1.0 - (np.var(x - xhat) / denom)

def split_trials_like_matlab(trial_ids, seed=42, test_frac=0.1):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(trial_ids)) > test_frac
    train_idx = np.where(mask)[0]
    test_idx = np.where(~mask)[0]
    return train_idx, test_idx

def train_model_on_sequences(args, X_train, tvec, latent_dim):
    device = get_device()
    model = ODEVAE(
        n_neurons=X_train.shape[-1],
        latent_dim=latent_dim,
        num_experts=args.num_experts,
        decoder_type=args.decoder_type,
        k_neighbors=getattr(args, "k_neighbors", 16),
        dec_num_experts=getattr(args, "dec_num_experts", 4),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tvec_t = torch.from_numpy(tvec).to(device)
    loader = td.DataLoader(SeqDataset(X_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    epochs = getattr(args, "r2_sweep_epochs", args.epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb in loader:
            xb = xb.to(device)
            if getattr(args, "baseline_correct", False):
                xb = baseline_correct_torch(xb)
            opt.zero_grad()
            xhat, mu, logvar, z_traj, zdiff = model(xb, tvec_t)
            if args.kl_warmup_epochs > 0:
                beta = args.beta * min(1.0, epoch / args.kl_warmup_epochs)
            else:
                beta = args.beta
            transition_loss = None
            lambda_transition = getattr(args, "lambda_transition", 0.0)
            if getattr(args, "lambda_transition_warmup_epochs", 0) > 0:
                lambda_transition *= min(1.0, epoch / args.lambda_transition_warmup_epochs)
            if lambda_transition > 0:
                dx_hat = xhat[:, 1:, :] - xhat[:, :-1, :]
                dx_true = xb[:, 1:, :] - xb[:, :-1, :]
                if getattr(args, "transition_landmark_count", 0) > 0:
                    trial_features = xb.mean(dim=1).detach().cpu().numpy()
                    lm_idx = greedy_landmarks(
                        trial_features,
                        k=min(args.transition_landmark_count, xb.shape[0])
                    )
                    dx_hat = dx_hat[lm_idx]
                    dx_true = dx_true[lm_idx]
                transition_loss = torch.mean((dx_hat - dx_true) ** 2)
            lle_loss = None
            lambda_lle = getattr(args, "lambda_lle", 0.0)
            if lambda_lle > 0:
                lle_loss = compute_lle_loss(
                    z_traj,
                    k=getattr(args, "lle_k", 8),
                    max_points=getattr(args, "lle_max_points", 256),
                    temperature=getattr(args, "lle_temperature", 0.1),
                )
            loss, *_ = vae_loss(
                xhat,
                xb,
                mu,
                logvar,
                zdiff,
                beta=beta,
                lambda_smooth=args.lambda_smooth,
                transition_loss=transition_loss,
                lambda_transition=lambda_transition,
                lle_loss=lle_loss,
                lambda_lle=lambda_lle,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    return model

def predict_sequences(model, X, tvec, args):
    device = get_device()
    model.eval()
    preds = []
    tvec_t = torch.from_numpy(tvec).to(device)
    loader = td.DataLoader(SeqDataset(X), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            if getattr(args, "baseline_correct", False):
                xb = baseline_correct_torch(xb)
            xhat, *_ = model(xb, tvec_t)
            preds.append(xhat.cpu().numpy())
    return np.concatenate(preds, axis=0)

def run_r2_sweep(args):
    npz = np.load(PATHS["data"])
    X_raw, tvec_np, trial_ids = make_sequences_raw(
        npz,
        trial_len_s=args.trial_len_s,
        fps=args.fps,
        drop_first_trials=args.drop_first_trials,
        min_frames=10,
    )
    dims = getattr(args, "r2_sweep_dims", list(range(1, 11)))
    if isinstance(dims, str):
        dims = [int(x.strip()) for x in dims.split(",") if x.strip()]
    elif not isinstance(dims, list):
        dims = [int(dims)]
    repeats = getattr(args, "r2_sweep_repeats", 3)
    seed = getattr(args, "r2_sweep_seed", 42)

    results = {}
    for d in dims:
        results[int(d)] = {"overall": [], "trials": []}
        for rep in range(repeats):
            train_idx, test_idx = split_trials_like_matlab(trial_ids, seed=seed + rep, test_frac=0.1)
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            X_train_raw = X_raw[train_idx]
            X_test_raw = X_raw[test_idx]
            X_train_norm, mu, sd = normalize_sequences(X_train_raw)
            X_test_norm, _, _ = normalize_sequences(X_test_raw, mu, sd)

            if getattr(args, "baseline_correct", False):
                X_train_norm = baseline_correct_np(X_train_norm)
                X_test_norm = baseline_correct_np(X_test_norm)
                X_test_eval = baseline_correct_np(X_test_raw)
                xhat_bias = False
            else:
                X_test_eval = X_test_raw
                xhat_bias = True

            n_components = getattr(args, "pca_dim", 0)
            if isinstance(n_components, int) and n_components > 0:
                n_components = n_components
            else:
                n_components = getattr(args, "pca_variance", 0.95)
            pca = PCA(n_components=n_components, svd_solver="full")
            flat_train = X_train_norm.reshape(-1, X_train_norm.shape[-1])
            pca.fit(flat_train)

            X_train_pca = pca.transform(flat_train).reshape(X_train_norm.shape[0], X_train_norm.shape[1], -1)
            X_test_pca = pca.transform(X_test_norm.reshape(-1, X_test_norm.shape[-1])).reshape(X_test_norm.shape[0], X_test_norm.shape[1], -1)

            model = train_model_on_sequences(args, X_train_pca, tvec_np, latent_dim=int(d))
            xhat_pca = predict_sequences(model, X_test_pca, tvec_np, args)

            xhat_norm = pca.inverse_transform(xhat_pca.reshape(-1, xhat_pca.shape[-1])).reshape(X_test_pca.shape[0], X_test_pca.shape[1], -1)
            if xhat_bias:
                xhat_raw = xhat_norm * sd + mu
            else:
                xhat_raw = xhat_norm * sd

            r2_all = r2_var_explained(X_test_eval, xhat_raw)
            results[int(d)]["overall"].append(r2_all)
            for i in range(X_test_eval.shape[0]):
                r2_i = r2_var_explained(X_test_eval[i], xhat_raw[i])
                if not np.isnan(r2_i):
                    results[int(d)]["trials"].append(r2_i)

    # plot (MATLAB-style)
    plt.figure(figsize=(6, 4))
    for d in dims:
        d = int(d)
        ys = results[d]["trials"]
        plt.scatter([d] * len(ys), ys, facecolors="none", edgecolors="k", alpha=0.6, s=30)
        if results[d]["overall"]:
            plt.scatter(d, np.mean(results[d]["overall"]), color="red", s=40, zorder=3)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Crossval R²")
    plt.xlim(0, max(dims) + 1)
    plt.ylim(0, 1)
    out_path = os.path.join(PATHS["out_dir"], "r2_sweep.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote R² sweep plot → {out_path}")
    return results

def greedy_landmarks(X, k=200):
    """
    Greedy selection of k landmarks from X to maximize coverage.
    Equivalent to MIND's greedyvq.m
    """
    X = np.asarray(X)
    n = len(X)
    if n <= k:
        return np.arange(n)
    landmarks = [np.random.randint(n)]
    for _ in range(k - 1):
        d = pairwise_distances(X, X[landmarks]).min(axis=1)
        landmarks.append(np.argmax(d))
    return np.array(landmarks)

class SeqDataset(td.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self): 
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i] # [L, N]

#____________________model__________________#
class Encoder(nn.Module):
    def __init__(self, n_in, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
    def forward(self, x0): # x0: [B, N]
        h = self.net(x0)
        return self.mu(h), self.logvar(h)
    
class MoELatentODEFunc(nn.Module):
    def __init__(self, latent_dim, num_experts=4, hidden=128):
        super().__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim

        # ----- gating network -----
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_experts)   # logits
        )

        # ----- expert ODE vector fields -----
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, latent_dim),
            )
            for _ in range(num_experts)
        ])

        # layernorm for stability
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, t, z):
        """
        z: [B, D]
        returns z': [B, D]
        """

        # ----- gating -----
        logits = self.gate(z)                       # [B, num_experts]
        weights = torch.softmax(logits, dim=-1)     # convex combination

        # ----- compute each expert’s derivative -----
        # result: list of E tensors, each [B, D]
        expert_outs = torch.stack(
            [expert(z) for expert in self.experts],  # [E, B, D]
            dim=0
        )

        # ----- combine using weights -----
        # weights: [B, E] → reshape to [E, B, 1] to match broadcasting
        weights_expanded = weights.transpose(0,1).unsqueeze(-1)  # [E, B, 1]

        dz = (weights_expanded * expert_outs).sum(dim=0)         # [B, D]

        # layernorm for stability (important!)
        return self.ln(dz)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        h = 128
        self.f1 = nn.Sequential(nn.Linear(latent_dim, h), nn.SiLU(),
                                nn.Linear(h, h), nn.SiLU())
        self.f2 = nn.Linear(h, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, t, z):
        h = self.f1(z)
        dz = self.f2(h)
        # simple residual-normalized field for stability
        return self.ln(dz)

class LocalAttentionDecoder(nn.Module):
    """
    A decoder that reconstructs each neuron using a weighted combination
    of nearby neurons in latent space — approximating LLE-like locality.

    For each neuron i:
        x_i(t) = f( concat[z_t, sum_j w_ij * z_t] )
    """
    def __init__(self, latent_dim, n_neurons, k_neighbors=16, hidden=256):
        super().__init__()
        self.n_neurons = n_neurons
        self.k = min(k_neighbors, n_neurons)

        # Learnable embeddings to define similarity between neurons
        self.emb = nn.Embedding(n_neurons, latent_dim)

        # Decoder MLP
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_traj):
        """
        z_traj: [B, L, D]
        Output: [B, L, N]
        """
        B, L, D = z_traj.shape
        device   = z_traj.device
        N        = self.n_neurons

        # --------------------------------------------------------
        # 1. Compute pairwise similarity between neuron embeddings
        # --------------------------------------------------------
        emb = self.emb.weight                     # [N, D]
        sim = emb @ emb.t()                       # [N, N]

        # --------------------------------------------------------
        # 2. Get top-k nearest neighbors for each neuron
        # --------------------------------------------------------
        _, idx = torch.topk(sim, k=self.k+1, dim=-1)
        idx = idx[:, 1:]                           # remove self-neuron
        # idx: [N, k]

        # --------------------------------------------------------
        # 3. Compute attention weights for neighbors
        # --------------------------------------------------------
        neigh_emb = emb[idx]                      # [N, k, D]

        # attention score for neighbor j of neuron i
        # dot product: (i·j)
        attn = torch.softmax(
            (emb.unsqueeze(1) * neigh_emb).sum(-1),   # [N, k]
            dim=-1
        )

        # --------------------------------------------------------
        # 4. Compute local latent: weighted sum of neighbor latents
        # --------------------------------------------------------
        # z_traj: [B, L, D]
        # expand to: [B, L, 1, D] → [B, L, N, D]
        z_expanded = z_traj[:, :, None, :].expand(B, L, N, D)

        # build neighbor latent tensor: [B, L, N, k, D]
        z_neigh = z_traj[:, :, None, None, :].expand(B, L, N, self.k, D)

        # attn: [N, k] → expand: [1, 1, N, k, 1]
        attn_expanded = attn[None, None, :, :, None]

        # local aggregate: [B, L, N, D]
        z_loc = (attn_expanded * z_neigh).sum(dim=3)

        # --------------------------------------------------------
        # 5. Decode each neuron: concat(global, local)
        # --------------------------------------------------------
        dec_in = torch.cat([z_expanded, z_loc], dim=-1)  # [B, L, N, 2D]

        out = self.net(dec_in.reshape(B*L*N, 2*D)).view(B, L, N)
        return out
    
class NeuronAwareDecoder(nn.Module):
    def __init__(self, latent_dim, n_neurons, emb_dim=16, hidden=256):
        """
        latent_dim: dim of z_t
        n_neurons:  number of output channels (N)
        emb_dim:    size of per-neuron embedding vector
        hidden:     hidden width of the MLP
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.emb = nn.Embedding(n_neurons, emb_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # predict one value per neuron
        )

    def forward(self, z_traj):  # z_traj: [B, L, D]
        B, L, D = z_traj.shape
        device = z_traj.device
        N = self.n_neurons

        # neuron indices 0..N-1
        neuron_idx = torch.arange(N, device=device)         # [N]
        neuron_emb = self.emb(neuron_idx)                   # [N, E]

        # broadcast latent trajectory over neurons
        # z_exp: [B, L, N, D]
        z_exp = z_traj.unsqueeze(2).expand(B, L, N, D)

        # broadcast embeddings over (B, L)
        # e_exp: [B, L, N, E]
        e_exp = neuron_emb.view(1, 1, N, -1).expand(B, L, N, neuron_emb.shape[1])

        # combine and run through MLP
        inp = torch.cat([z_exp, e_exp], dim=-1)             # [B, L, N, D+E]
        inp = inp.reshape(B * L * N, D + neuron_emb.shape[1])

        out = self.net(inp).view(B, L, N)                   # [B, L, N]
        return out

class MoEDecoder(nn.Module):
    """
    Decoder MoE:
      - E shared decoder experts: f_e(z_t) -> R^N
      - Each neuron i has its own softmax over experts w_{i,e}.
      - Output: x(t) = sum_e w[:, e] * f_e(z_t).
    """
    def __init__(self, latent_dim, n_neurons, num_experts=4, hidden=256):
        super().__init__()
        self.num_experts = num_experts
        self.n_neurons = n_neurons

        # E experts: each maps latent_dim -> n_neurons
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_neurons)
            )
            for _ in range(num_experts)
        ])

        # Per-neuron logits over experts: [N, E]
        # Softmax over dim=-1 gives w_{i,e}
        self.neuron_logits = nn.Parameter(
            torch.zeros(n_neurons, num_experts)
        )

    def forward(self, z_traj):  # [B, L, D]
        B, L, D = z_traj.shape
        E = self.num_experts
        N = self.n_neurons

        # compute experts' outputs
        z_flat = z_traj.reshape(B * L, D)  # [B*L, D]
        expert_outs = []

        for e in self.experts:
            y = e(z_flat).view(B, L, N)    # [B, L, N]
            expert_outs.append(y)

        # [E, B, L, N]
        expert_outs = torch.stack(expert_outs, dim=0)

        # neuron-specific mixture weights: [N, E] -> softmax over experts
        w = torch.softmax(self.neuron_logits, dim=-1)       # [N, E]
        # reshape for broadcasting with [E, B, L, N]
        w = w.permute(1, 0).view(E, 1, 1, N)                # [E, 1, 1, N]

        # weighted sum over experts
        y = (expert_outs * w).sum(dim=0)                    # [B, L, N]
        return y

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),      # ← extra hidden layer
        nn.ReLU(),
        nn.Linear(512, n_out)
)
    def forward(self, z_traj): # [B, L, D]
        B, L, D = z_traj.shape
        x = self.net(z_traj.reshape(B*L, D)) # [B*L, N]
        return x.reshape(B, L, -1)
    
class ODEVAE(nn.Module):
    def __init__(self, n_neurons, latent_dim, num_experts, decoder_type="mlp", k_neighbors=16, dec_num_experts = 4):
        super().__init__()
        self.enc = Encoder(n_neurons, latent_dim)
        self.odefunc = MoELatentODEFunc(latent_dim, num_experts=num_experts)
        # add noise during training to prevent the ODE from overfitting tiny 
        # geometric details
        # self.latent_noise_std = 0.0

        # select decoder
        if decoder_type.lower() == "mlp":
            self.dec = Decoder(latent_dim, n_neurons)

        elif decoder_type.lower() == "neuronaware":
            self.dec = NeuronAwareDecoder(latent_dim, n_neurons)

        elif decoder_type.lower() == "localattn":
            self.dec = LocalAttentionDecoder(
                latent_dim=latent_dim,
                n_neurons=n_neurons,
                k_neighbors=k_neighbors
            )
        elif decoder_type.lower() == "moe":
            self.dec = MoEDecoder(
                latent_dim=latent_dim,
                n_neurons=n_neurons,
                num_experts=dec_num_experts,
                hidden=256
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def _integrate_latent(self, z0, tvec, method="dopri5"):
        """
        Integrate latent dynamics.
        method: "rk4" (fixed step) or "dopri5" (adaptive).
        """
        if method == "rk4":
            step = (tvec[1] - tvec[0]).abs().item()
            safe_step = step / 2.0

            z_traj = odeint(
                self.odefunc,
                z0,
                tvec,
                method ="rk4",
                options={"step_size": safe_step}
            )
            return z_traj
        z_traj = odeint(
            self.odefunc,
            z0,
            tvec,
            method="dopri5",
            rtol=1e-3,
            atol=1e-4
        )
        return z_traj

    def forward(self, x_seq, tvec):
        """
        x_seq: [B, L, N]
        tvec:  [L] strictly increasing (float tensor)
        """
        B, L, N = x_seq.shape
        x0 = x_seq[:, 0, :]                    # [B, N]
        mu, logvar = self.enc(x0)              # [B, D], [B, D]
        z0 = self.reparam(mu, logvar).float()
        tvec = tvec.float()

        z_traj = self._integrate_latent(z0, tvec, method="dopri5")

        z_traj = z_traj.permute(1, 0, 2).contiguous()                 # [B, L, D]

        # finite-difference for smoothness penalty
        dt = (tvec[1:] - tvec[:-1]).view(1, -1, 1)                    # [1, L-1, 1]
        zdiff = (z_traj[:, 1:, :] - z_traj[:, :-1, :]) / dt

        xhat = self.dec(z_traj)                                       # [B, L, N]
        return xhat, mu, logvar, z_traj, zdiff
    
#___________________loss_________________#
def compute_lle_loss(z_traj, k=8, max_points=256, temperature=0.1):
    # soft LLE: reconstruct each point from neighbors using softmax weights
    z_flat = z_traj.reshape(-1, z_traj.shape[-1])
    n = z_flat.shape[0]
    if n <= k:
        return z_traj.new_tensor(0.0)
    if n > max_points:
        idx = torch.randperm(n, device=z_traj.device)[:max_points]
        z_flat = z_flat[idx]
        n = z_flat.shape[0]
    dists = torch.cdist(z_flat, z_flat)
    eye = torch.eye(n, device=z_traj.device, dtype=torch.bool)
    dists = dists.masked_fill(eye, float("inf"))
    knn_dist, knn_idx = torch.topk(dists, k=k, largest=False)
    neigh = z_flat[knn_idx]
    weights = torch.softmax(-knn_dist / max(temperature, 1e-6), dim=-1)
    recon = (weights.unsqueeze(-1) * neigh).sum(dim=1)
    return torch.mean((z_flat - recon) ** 2)

def vae_loss(xhat, x, mu, logvar, zdiff, beta=1.0, lambda_smooth=0.0, transition_loss=None, lambda_transition=0.0, lle_loss=None, lambda_lle=0.0):
    # --- Safety clamp to prevent numerical overflow ---
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    recon = torch.mean((xhat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    smooth = torch.mean(zdiff**2) if lambda_smooth > 0 else x.new_tensor(0.0)
    if transition_loss is None or lambda_transition <= 0:
        transition = x.new_tensor(0.0)
    else:
        transition = transition_loss
    if lle_loss is None or lambda_lle <= 0:
        lle = x.new_tensor(0.0)
    else:
        lle = lle_loss
    total = recon + beta*kl + lambda_smooth*smooth + lambda_transition*transition + lambda_lle*lle
    return total, recon, kl, smooth, transition, lle


#_________________training_______________#
def train(args):
    device = get_device()
    print("Device:", device)

    #load data 
    npz = np.load(PATHS["data"])

    meta = {}

    # --- Step 1: PCA Preprocessing (MIND-style) ---
    roi = npz["roi"]
    if roi.shape[0] < roi.shape[1]:
        roi = roi.T  # [T, N]

    #   Keep 95% variance
    pca = PCA(n_components=0.95, svd_solver="full")
    roi_pca = pca.fit_transform(roi)
    print(f"PCA reduced {roi.shape[1]} → {roi_pca.shape[1]} dims ({pca.explained_variance_ratio_.sum():.2%} variance)")
    joblib.dump(pca, os.path.join(PATHS["out_dir"], "trained_pca.pkl"))
    # Replace ROI in npz-like structure
    npz_mod = dict(npz)
    npz_mod["roi"] = roi_pca
    npz = npz_mod
    X, tvec_np, meta = make_sequences(
        npz, 
        trial_len_s=args.trial_len_s,
        fps=args.fps, 
        drop_first_trials=args.drop_first_trials,
        min_frames=10
    ) # X: [B, L, N]

    # --- Normalize time vector to [0,1] for numerical stability ---
    tvec_np = tvec_np / tvec_np[-1]

    

    # --- Step 2: Landmark Subsampling (optional) ---
    if getattr(args, "landmark_count", 0) > 0:
        print(f"Selecting {args.landmark_count} landmark trials (greedy coverage)...")
        # Flatten trials along time for selection
        X_flat = X.reshape(-1, X.shape[-1])
        lm_idx = greedy_landmarks(X_flat, k=args.landmark_count)
        X = X[lm_idx % X.shape[0]]  # map back to batch level
        print(f"Subsampled to {X.shape[0]} sequences.")

    B, L, N = X.shape
    print(f"Built sequences: B={B}, :={L}, N={N}")

    # train/val split (hold out last K trials)
    holdout = min(args.holdout_trials, max(1, B //5))
    X_train = X[:-holdout]
    X_val = X[-holdout:]

    train_loader = td.DataLoader(SeqDataset(X_train), batch_size = args.batch_size, shuffle=True, drop_last = True)
    val_loader = td.DataLoader(SeqDataset(X_val), batch_size = args.batch_size, shuffle=False)

    model = ODEVAE(
    n_neurons=N,
    latent_dim=args.latent_dim,
    num_experts=args.num_experts,
    decoder_type=args.decoder_type,
    k_neighbors=getattr(args, "k_neighbors", 16),
    dec_num_experts=getattr(args, "dec_num_experts", 4)
        ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tvec = torch.from_numpy(tvec_np).to(device)

    best_val = math.inf
    os.makedirs(args.out_dir, exist_ok = True)

    # a global break safeguard, stops training if NaNs are detected
    nan_flag = False

    for epoch in range(1, args.epochs+1):
        #_____ train 
        model.train()
        tl, tr, tk, ts, tt, tll = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for xb in train_loader:
            xb = xb.to(device) # [B, L, N]

            # this is trial-wise baseline correction, getting rid of the first 5 frames
            # so the latent doesn't waste space capacity on slow drifs/offsets
            if getattr(args, "baseline_correct", False):
                # subtract per-trial baseline over first 5 frames
                baseline = xb[:, :5, :].mean(dim=1, keepdim=True)  # [B, 1, N]
                xb = xb - baseline

            opt.zero_grad()
            xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
            # KL warmup!
            if args.kl_warmup_epochs > 0:
                beta = args.beta * min(1.0, epoch /args.kl_warmup_epochs)
            else:
                beta = args.beta
            transition_loss = None
            lambda_transition = getattr(args, "lambda_transition", 0.0)
            if getattr(args, "lambda_transition_warmup_epochs", 0) > 0:
                lambda_transition *= min(1.0, epoch / args.lambda_transition_warmup_epochs)
            if lambda_transition > 0:
                # match decoded transition dynamics without re-decoding z_pred
                dx_hat = xhat[:, 1:, :] - xhat[:, :-1, :]
                dx_true = xb[:, 1:, :] - xb[:, :-1, :]
                if getattr(args, "transition_landmark_count", 0) > 0:
                    trial_features = xb.mean(dim=1).detach().cpu().numpy()
                    lm_idx = greedy_landmarks(
                        trial_features,
                        k=min(args.transition_landmark_count, xb.shape[0])
                    )
                    dx_hat = dx_hat[lm_idx]
                    dx_true = dx_true[lm_idx]
                transition_loss = torch.mean((dx_hat - dx_true) ** 2)
            lle_loss = None
            lambda_lle = getattr(args, "lambda_lle", 0.0)
            if lambda_lle > 0:
                lle_loss = compute_lle_loss(
                    z_traj,
                    k=getattr(args, "lle_k", 8),
                    max_points=getattr(args, "lle_max_points", 256),
                    temperature=getattr(args, "lle_temperature", 0.1),
                )
            loss, rec, kl, sm, trn, lle = vae_loss(
                xhat,
                xb,
                mu,
                logvar,
                zdiff,
                beta=beta,
                lambda_smooth=args.lambda_smooth,
                transition_loss=transition_loss,
                lambda_transition=lambda_transition,
                lle_loss=lle_loss,
                lambda_lle=lambda_lle,
            )
            
            # ---- NaN check ----
            if torch.isnan(loss):
                print(f"[epoch {epoch}] NaN detected — stopping training early.")
                nan_flag = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item(); tr += rec.item(); tk+= kl.item(); ts += sm.item(); tt += trn.item(); tll += lle.item()
        nb = len(train_loader)
        print(f"[{epoch:03d}] train loss {tl/nb:.5f} | recon { tr/nb:.5f} | kl {tk/nb:.5f} | smooth {ts/nb:.5f} | trans {tt/nb:.5f} | lle {tll/nb:.5f} | beta {beta:.3f}")

        if nan_flag:
            break

        # ______val
        model.eval()
        with torch.no_grad(): 
            vl, vr, vk, vs, vt, vlle, r2_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            n_batches = 0
            for xb in val_loader:
                # this is trial-wise baseline correction, getting rid of the first 5 frames
                # so the latent doesn't waste space capacity on slow drifs/offsets
                if getattr(args, "baseline_correct", False):
                    # subtract per-trial baseline over first 5 frames
                    baseline = xb[:, :5, :].mean(dim=1, keepdim=True)  # [B, 1, N]
                    xb = xb - baseline
                xb = xb.to(device)
                xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
                transition_loss = None
                lambda_transition = getattr(args, "lambda_transition", 0.0)
                if getattr(args, "lambda_transition_warmup_epochs", 0) > 0:
                    lambda_transition *= min(1.0, epoch / args.lambda_transition_warmup_epochs)
                if lambda_transition > 0:
                    dx_hat = xhat[:, 1:, :] - xhat[:, :-1, :]
                    dx_true = xb[:, 1:, :] - xb[:, :-1, :]
                    if getattr(args, "transition_landmark_count", 0) > 0:
                        trial_features = xb.mean(dim=1).detach().cpu().numpy()
                        lm_idx = greedy_landmarks(
                            trial_features,
                            k=min(args.transition_landmark_count, xb.shape[0])
                        )
                        dx_hat = dx_hat[lm_idx]
                        dx_true = dx_true[lm_idx]
                    transition_loss = torch.mean((dx_hat - dx_true) ** 2)
                lle_loss = None
                lambda_lle = getattr(args, "lambda_lle", 0.0)
                if lambda_lle > 0:
                    lle_loss = compute_lle_loss(
                        z_traj,
                        k=getattr(args, "lle_k", 8),
                        max_points=getattr(args, "lle_max_points", 256),
                        temperature=getattr(args, "lle_temperature", 0.1),
                    )
                loss, rec, kl, sm, trn, lle = vae_loss(
                    xhat,
                    xb,
                    mu,
                    logvar,
                    zdiff,
                    beta=args.beta,
                    lambda_smooth=args.lambda_smooth,
                    transition_loss=transition_loss,
                    lambda_transition=lambda_transition,
                    lle_loss=lle_loss,
                    lambda_lle=lambda_lle,
                )
                vl += loss.item(); vr += rec.item(); vk += kl.item(); vs += sm.item(); vt += trn.item(); vlle += lle.item()

                # --- R² computation per batch ---
                r2_batch = compute_r2(xb.cpu(), xhat.cpu())
                if not np.isnan(r2_batch):
                    r2_total += r2_batch
                n_batches += 1

            nbv = len(val_loader)
            mean_r2 = r2_total / max(1, n_batches)
            print(f"      valid loss {vl/nbv:.5f} | recon {vr/nbv:.5f} | kl {vk/nbv:.5f} | smooth {vs/nbv:.5f} | trans {vt/nbv:.5f} | lle {vlle/nbv:.5f} | R² {mean_r2:.4f}")

            if vl/nbv < best_val: 
                    best_val = vl/nbv
                    ckpt = os.path.join(PATHS["out_dir"], "ode_vae_best.pt")

                    # obtains the hash of the input file for logging
                    data_hash = compute_file_sha(PATHS["data"])

                    # verifies no nans, then saaves the model 
                    if not nan_flag and vl/nbv < best_val:
                        torch.save({
                            "state_dict": model.state_dict(),
                            "tvec": tvec_np,
                            "meta": meta,
                            "args": vars(args),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "git_commit": os.popen("git rev-parse HEAD").read().strip() or "unknown",
                            "data_hash": data_hash,
                        }, ckpt)

        # print("  saved best model to", ckpt)

        # quick preview image (first batch first trial)
        try:
            xb = next(iter(val_loader)).to(device)
            xhat, *_ = model(xb, tvec)
            xb_np   = xb[0].detach().cpu().numpy()      # [L, N]
            xhat_np = xhat[0].detach().cpu().numpy()
            # plot mean across neurons for a quick sanity check
            plt.figure(figsize=(8,3))
            plt.plot(xb_np.mean(axis=1), label="GT mean")
            plt.plot(xhat_np.mean(axis=1), label="Recon mean", alpha=0.8)
            plt.legend(); plt.title("Validation mean activity (GT vs Recon)")
            out_png = PATHS["preview"]
            plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
            # print(f"      wrote {out_png}")
        except Exception as e:
            print("      (preview plot skipped:", e, ")")

            # --- Reconstruction accuracy over time (R² and MSE per timestep) ---
        '''
        try:
            xb = next(iter(val_loader)).to(device)
            xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
            xb_np = xb[0].detach().cpu().numpy()      # [L, N]
            xhat_np = xhat[0].detach().cpu().numpy()  # [L, N]

            # Compute R² and MSE per time step
            r2_t = []
            mse_t = []
            for t in range(xb_np.shape[0]):
                r2_t.append(r2_score(xb_np[t], xhat_np[t]))
                mse_t.append(np.mean((xb_np[t] - xhat_np[t])**2))

            r2_t = np.array(r2_t)
            mse_t = np.array(mse_t)

            time_axis = np.arange(len(r2_t)) / args.fps  # convert frames → seconds

            fig, ax1 = plt.subplots(figsize=(8, 4))
            color_r2 = 'tab:blue'
            color_mse = 'tab:red'

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('R²', color=color_r2)
            ax1.plot(time_axis, r2_t, color=color_r2, label='R²(t)')
            ax1.tick_params(axis='y', labelcolor=color_r2)
            ax1.set_ylim(-1, 1.1)

            ax2 = ax1.twinx()
            ax2.set_ylabel('MSE', color=color_mse)
            ax2.plot(time_axis, mse_t, color=color_mse, linestyle='--', label='MSE(t)')
            ax2.tick_params(axis='y', labelcolor=color_mse)

            plt.title('Reconstruction Accuracy Over Time')
            fig.tight_layout()
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(PATHS["out_dir"], "recon_accuracy_over_time.png"), dpi=160)
            plt.close()
            print("      wrote reconstruction accuracy plot → recon_accuracy_over_time.png")

        except Exception as e:
            print("      (reconstruction accuracy plot skipped:", e, ")")
        '''

    # --- Step 3: Latent manifold embedding (MIND-style) ---
    try:
        xb = next(iter(val_loader)).to(device)
        xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
        z_flat = z_traj[0].detach().cpu().numpy()  # [L, D]

        # Compute pairwise distance matrix and apply MDS
        from sklearn.manifold import MDS
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(z_flat)
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=0, n_init = 1)
        embed = mds.fit_transform(D)

        # Plot MDS embedding (color by time)
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        t = np.arange(len(embed))
        p = ax.scatter(embed[:,0], embed[:,1], embed[:,2], c=t, cmap="viridis", s=8)
        fig.colorbar(p, ax=ax, label="Time")
        ax.set_title("Latent Manifold Embedding (MIND-style)")
        plt.tight_layout()
        out_path = os.path.join(PATHS["out_dir"], "latent_manifold_mds.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"      wrote {out_path}")
    except Exception as e:
        print("      (MIND manifold plot skipped:", e, ")")

    print("done.")
    # Check out decoder bias terms

    for name, param in model.dec.named_parameters():
        if 'bias' in name:
            print(name, param.data.mean().item())

    final_metrics = {
    "recon": vr / nbv,
    "kl": vk / nbv,
    "smooth": vs / nbv,
    "r2": mean_r2,
    "lle": vlle / nbv,
    }
    # --- Log run metadata to JSON file --- #
    run_metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "git_commit": os.popen("git rev-parse HEAD").read().strip() or "unknown",
    "hyperparameters": vars(args),
    "final_metrics": {
        "best_val_loss": best_val,
        "final_r2": mean_r2,
        "recon": vr / nbv,
        "kl": vk / nbv,
        "smooth": vs / nbv,
        "transition": vt / nbv,
        "lle": vlle / nbv,
        "data_hash": data_hash,
    }
    }

    json_path = os.path.join(args.out_dir, "run_metadata.json")
    with open(json_path, "w") as f:
        json.dump(to_jsonable(final_metrics), f, indent=2)
        print(f"Saved run metadata → {json_path}")

    torch.save(final_metrics, PATHS["final_metrics"])
    print(f"Final R²: {mean_r2:.4f}")
    return best_val, mean_r2



#__________________main_____________#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(SRC_DIR, "config.txt"), help="Path to config file")
    args_cli = ap.parse_args()

    # load defaults from file
    if os.path.exists(args_cli.config):
        print(f"Loading configuration from {args_cli.config}")
        cfg = load_config_from_txt(args_cli.config)
    else:
        raise FileNotFoundError(f"Config file not found: {args_cli.config}")

    # convert dict to namespace (so it can be called from train(args))
    class Struct:
        def __init__(self, **entries): self.__dict__.update(entries)
    args = Struct(**cfg)

    '''
    #----------------OPTIONAL: Seed Sweep ----------------#
    # What this does: sweeps across multiple random seeds to find the best performing model.

    seed_list = [1, 42, 1337, 2025, 777]
    results = []

    for seed in seed_list:
        print(f"\n===== Running seed {seed} =====")
        set_seed(seed)
        start_time = datetime.datetime.now()
        best_val, mean_r2 = train(args)
        end_time = datetime.datetime.now()
        results.append((seed, best_val, mean_r2))

    # ========== SUMMARY ==========
    print("\n===== Seed Sweep Summary =====")
    for seed, val, r2 in results:
        print(f"Seed {seed:4d} → R²={r2:.4f} | best val loss={val:.5f}")

    best_run = max(results, key=lambda x: x[2])
    print(f"\nBest seed: {best_run[0]} → R²={best_run[2]:.4f}")
    '''

    if getattr(args, "r2_sweep_enabled", False):
        run_r2_sweep(args)
        raise SystemExit(0)

    seed = 1  # using the best seed from the results from the sweep above 
    results = []
    set_seed(seed)
    start_time = datetime.datetime.now()
    best_val, mean_r2 = train(args)
    end_time = datetime.datetime.now()
    results.append((seed, best_val, mean_r2))

    # ========== LOG RESULTS ========== #
    log_file = PATHS["training_log"]
    result_lines = []

    # read existing logs if any
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            result_lines = f.readlines()

    # add new entry at the top
    new_entry = (
    f"=== Run at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    f"Commit: {os.popen('git rev-parse HEAD').read().strip() or 'unknown'}\n"
    f"Data: {PATHS['data']}\n"
    f"Latent dim: {args.latent_dim} | Epochs: {args.epochs} | LR: {args.lr}\n"
    f"Batch size: {args.batch_size} | Beta: {args.beta} | Smooth λ: {args.lambda_smooth}\n"
    f"Holdout: {args.holdout_trials} | KL warmup: {args.kl_warmup_epochs}\n"
    f"Final validation loss: {best_val:.5f}\n"
    f"Final R² value: {mean_r2:.4f}\n"
    f"Saved model: {os.path.join(PATHS['out_dir'], 'ode_vae_best.pt')}\n"
    f"---------------------------------------------\n"
    )
    result_lines.insert(0, new_entry)

    # write back to file
    with open(log_file, "w") as f:
        f.writelines(result_lines)

    print(f"Results logged to {log_file}")
