"""
v2_neural_vae.py — Switching Neural ODE VAE (mixture/gated latent dynamics)

Pipeline:
- Same overall sequence construction approach as v1 (trial grouping + resampling).
- Typically uses z-scoring and time normalization; supports MPS-safe integration settings.

Model:
- Encoder: MLP x(t0) -> (mu, logvar).
- Latent dynamics: SwitchingLatentODEFunc = K candidate vector fields f_k(z) with a learned gating network g(z);
  dz/dt = sum_k softmax(g(z))_k * f_k(z).
- Decoder: MLP z(t) -> x̂(t).

Training objective:
- Reconstruction MSE + beta * KL + lambda_smooth * latent smoothness (finite-difference on z(t)).

Intended difference vs v1:
- Adds “multiple regimes” dynamics in latent space (analogous to multiple local models).
"""

# note: using float32 inputs and added some code to get apple's MPS backend to work,
# becuase torchdiffeq likes to create float64 tensors for tolerances (rtol, atol) which MPS doesn't support. 

# Written by Kathleen Higgins
# Worked as of 2025-09-10
# src/train_neural_ode_vae.py
import os, math, argparse, datetime
import sys
import atexit
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

PATHS = {
    # "data":          os.path.join(SRC_DIR, "random_walk_data", "synthetic_rat_data.npz"),
    "data":          os.path.join(SRC_DIR, "npz_e65_data", "E65_data.npz"),
    "out_dir":       os.path.join(SRC_DIR, "pt_files"),
    "final_metrics": os.path.join(SRC_DIR, "pt_files", "final_metrics.pt"),
    "preview":       os.path.join(SRC_DIR, "preview.png"),
    "training_log":  os.path.join(SRC_DIR, "training_results.txt"),
    "config":        os.path.join(SRC_DIR, "config.txt"),
    "run_output":    os.path.join(SRC_DIR, "script_output.txt"),
}
os.makedirs(PATHS["out_dir"], exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_run_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    def _close_log():
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_close_log)


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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
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

def make_sequences_raw(npz, trial_len_s=12.0, fps=10.0, min_frames=10):
    """
    Extract per-trial sequences without any PCA or z-scoring*.
    OUTPUT:
        X_raw: list of raw resampled trials (each [L, N])
        trial_ids: list of trial numbers in same order
        tvec: shared time vector [L]
    """
    roi  = npz["roi"]
    trial = npz["Trial"].astype(int)
    time  = npz["Time"].astype(float)

    if roi.shape[0] < roi.shape[1]:
        roi = roi.T  # [T, N]

    trials = np.unique(trial)
    groups = [np.where(trial == tr)[0] for tr in trials]

    L = int(round(trial_len_s * fps))
    t_dst = np.linspace(0, trial_len_s, L, dtype=np.float32)

    X_raw = []
    tr_list = []

    for tr, idx in zip(trials, groups):
        if len(idx) < min_frames:
            continue
        x_tr = roi[idx, :]                  # [Ts, N]
        t_tr = time[idx] - time[idx][0]     # reset trial start to 0

        # resample BEFORE normalization/PCA
        x_rs, _ = resample_sequence(x_tr, t_tr, L, t0=0.0, t1=trial_len_s)
        X_raw.append(x_rs)                  # [L, N]
        tr_list.append(tr)

    X_raw = np.stack(X_raw, axis=0).astype(np.float32)
    return X_raw, np.array(tr_list), t_dst.astype(np.float32)

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
    
class SwitchingLatentODEFunc(nn.Module):
    def __init__(self, latent_dim, K=3):
        super().__init__()
        self.K = K

        # gating in latent space
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, K)
        )

        # K ODE vector fields
        self.fields = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, latent_dim)
            )
            for _ in range(K)
        ])

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, t, z):
        g = torch.softmax(self.gate(z), dim=-1)  # [B, K]
        dz = 0
        for k in range(self.K):
            dz_k = self.fields[k](z)
            dz += g[:, k:k+1] * dz_k
        return self.norm(dz)
        
class Decoder(nn.Module):
    def __init__(self, latent_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_out)
        )

    def forward(self, z):
        B, L, D = z.shape
        return self.net(z.reshape(B*L, D)).reshape(B, L, -1)
    
class ODEVAE(nn.Module):
    def __init__(self, n_neurons, latent_dim):
        super().__init__()
        self.enc = Encoder(n_neurons, latent_dim)
        self.odefunc = SwitchingLatentODEFunc(latent_dim)
        self.dec = Decoder(latent_dim, n_neurons)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def _integrate_latent(self, z0, tvec, method="rk4"):
        """
        Integrate latent dynamics with a backend that avoids dt-underflow on MPS.
        method: "rk4" (fixed step, safest on MPS) or "dopri5" (adaptive).
        """
        if method == "rk4":
            # Use your uniform t grid spacing as the fixed step size.
            # Slightly smaller fixed-step for MPS to reduce stiffness-related NaNs
            step = (tvec[1] - tvec[0]).abs().item()
            safe_step = step / 2.0  # you can try /4 if a seed still blows up

            z_traj = odeint(
                self.odefunc,
                z0,
                tvec,
                method="rk4",
                options={"step_size": safe_step}
            )
            return z_traj

        # ---- Adaptive fallback (CPU/double if on MPS) ----
        use_cpu_double = (z0.device.type == "mps")
        if use_cpu_double:
            z0_cpu = z0.detach().to("cpu", dtype=torch.float64)
            t_cpu  = tvec.detach().to("cpu", dtype=torch.float64)
            odefunc_cpu = self.odefunc.to("cpu")
            z_traj_cpu = odeint(
                odefunc_cpu,
                z0_cpu,
                t_cpu,
                method="dopri5",
                rtol=1e-3,   # slightly looser than defaults
                atol=1e-4
            )
            # move results back to original device/dtype
            z_traj = z_traj_cpu.to(z0.device, dtype=z0.dtype)
            # return odefunc to original device
            self.odefunc.to(z0.device)
            return z_traj
        else:
            # GPU/CPU non-MPS path: adaptive dopri5 with modest tolerances
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

        # --- Use fixed-step RK4 by default on MPS to avoid dt underflow ---
        prefer_fixed = (z0.device.type == "mps")
        if prefer_fixed:
            z_traj = self._integrate_latent(z0, tvec, method="rk4")   # [L, B, D]
        else:
            z_traj = self._integrate_latent(z0, tvec, method="dopri5")

        z_traj = z_traj.permute(1, 0, 2).contiguous()                 # [B, L, D]

        # finite-difference for smoothness penalty
        dt = torch.clamp(tvec[1:] - tvec[:-1], min=1e-4).view(1, -1, 1)                # [1, L-1, 1]
        zdiff = (z_traj[:, 1:, :] - z_traj[:, :-1, :]) / dt

        xhat = self.dec(z_traj)                             # [B, L, N]
        return xhat, mu, logvar, z_traj, zdiff
    
#___________________loss_________________#
def vae_loss(xhat, x, mu, logvar, zdiff, beta=1.0, lambda_smooth=0.0):
    # --- Safety clamp to prevent numerical overflow ---
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    # weighted per-neuron loss accounting for sparse & bursty neurons
    eps = 1e-6

    # zero-inflated model
    mask_zero = (x == 0).float()
    mask_nonzero = 1 - mask_zero

    recon = torch.mean((xhat - x)**2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    smooth = torch.mean(zdiff**2) if lambda_smooth > 0 else x.new_tensor(0.0)
    return recon + beta*kl + lambda_smooth*smooth, recon, kl, smooth


#_________________training_______________#
def train(args):
    device = get_device()
    print("Device:", device)

    #load data 
    npz = np.load(PATHS["data"])

    # Compute data hash once at the start
    data_hash = compute_file_sha(PATHS["data"])

    meta = {}

    # --------------------------
    # 1. Extract raw ROI
    # --------------------------
    roi = npz["roi"]
    trial = npz["Trial"].astype(int)
    time  = npz["Time"].astype(float)

    if roi.shape[0] < roi.shape[1]:
        roi = roi.T  # [T, N]

    T, N_raw = roi.shape

    # --- Build raw fixed-length sequences FIRST ---
    X_raw, tr_list, tvec_np = make_sequences_raw(
        npz,
        trial_len_s=args.trial_len_s,
        fps=args.fps,
        min_frames=10
    )

    # --- Now split trials ---
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr_list))

    holdout = args.holdout_trials
    val_ids = perm[-holdout:]
    train_ids = perm[:-holdout]

    X_train_raw = X_raw[train_ids]  # [Btr, L, N]
    X_val_raw   = X_raw[val_ids]

    flat_train = X_train_raw.reshape(-1, X_train_raw.shape[-1])

    mu = flat_train.mean(axis=0, keepdims=True)
    sd = flat_train.std(axis=0, keepdims=True) + 1e-8

    X_train_std = (X_train_raw - mu) / sd
    X_val_std   = (X_val_raw   - mu) / sd

    roi_std = (roi - mu) / sd

    # --------------------------
    # 4. Train-only PCA
    # --------------------------
    pca = PCA(n_components=args.pca_dim, svd_solver="full")

    flat_train_std = X_train_std.reshape(-1, X_train_std.shape[-1])
    pca.fit(flat_train_std)

    joblib.dump(pca, os.path.join(PATHS["out_dir"], "trained_pca.pkl"))

    # transform per trial
    T, N_raw = roi.shape
    Btr, L, N_raw = X_train_std.shape
    Npca = pca.n_components_

    X_train_pca = pca.transform(
    X_train_std.reshape(-1, N_raw)
        ).reshape(Btr, L, Npca)
    X_val_pca = pca.transform(
    X_val_std.reshape(-1, N_raw)    
        ).reshape(len(X_val_std), L, Npca)

    # --------------------------
    # 5. Pass PCA’d data downstream
    # --------------------------
    # Instead of reconstructing sequences again using make_sequences (which no longer exists),
    # we simply use the PCA-transformed sequences we already built:

    X_train = X_train_pca
    X_val   = X_val_pca

    # Ensure tvec_np was already produced by make_sequences_raw()
    # so we keep it as-is.

    '''
    Commenting this out right now to not time normalize, so the ODE can see real-time spacing 
    # --- Normalize time vector to [0,1] for numerical stability ---
    tvec_np = tvec_np / tvec_np[-1]
    '''

    # --- Step 2: Landmark Subsampling (optional) ---

    B, L, Npca = X_train.shape
    print(f"Built sequences: B={B}, L={L}, N={Npca}")


    train_loader = td.DataLoader(SeqDataset(X_train), batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    val_loader   = td.DataLoader(SeqDataset(X_val), batch_size=args.batch_size,
                             shuffle=False)

    model = ODEVAE(n_neurons=Npca, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tvec = torch.from_numpy(tvec_np).to(device)

    best_val = math.inf
    os.makedirs(PATHS["out_dir"], exist_ok = True)

    # a global break safeguard, stops training if NaNs are detected
    nan_flag = False

    for epoch in range(1, args.epochs+1):
        #_____ train 
        model.train()
        tl, tr, tk, ts = 0.0, 0.0, 0.0, 0.0
        for xb in train_loader:
            xb = xb.to(device) # [B, L, N]
            opt.zero_grad()
            xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
            # KL warmup!
            if args.kl_warmup_epochs > 0:
                beta = args.beta * min(1.0, epoch /args.kl_warmup_epochs)
            else:
                beta = args.beta
            # ----- Per-trial training loss identical to validation -----

            # --- reconstruct raw space identical to validation ---
            Btr, Ltr, Nv_pca = xhat.shape
            xhat_flat = xhat.reshape(-1, Nv_pca).detach().cpu().numpy()
            xb_flat   = xb.reshape(-1, Nv_pca).detach().cpu().numpy()

            xhat_raw = pca.inverse_transform(xhat_flat).reshape(Btr, Ltr, -1)
            xb_raw   = pca.inverse_transform(xb_flat).reshape(Btr, Ltr, -1)

            # convert back to torch to compute loss on-device
            xhat_raw_t = torch.from_numpy(xhat_raw).float().to(device)
            xb_raw_t   = torch.from_numpy(xb_raw).float().to(device)


            per_trial_rec = torch.mean((xhat - xb) ** 2)
            per_trial_kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            if args.lambda_smooth > 0:
                per_trial_smooth = torch.mean(zdiff**2, dim=(1,2))
            else:
                per_trial_smooth = torch.zeros_like(per_trial_rec)

            per_trial_loss = per_trial_rec + beta * per_trial_kl + args.lambda_smooth * per_trial_smooth
            loss = per_trial_loss.mean()
            
            # ---- NaN check ----
            if torch.isnan(loss):
                print(f"[epoch {epoch}] NaN detected — stopping training early.")
                nan_flag = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            # accumulate using the correct variables
            tl += loss.item()
            tr += per_trial_rec.mean().item()
            tk += per_trial_kl.mean().item()
            ts += per_trial_smooth.mean().item()
        nb = len(train_loader)
        print(f"[{epoch:03d}] train loss {tl/nb:.5f} | recon { tr/nb:.5f} | kl {tk/nb:.5f} | smooth {ts/nb:.5f} | beta {beta:.3f}")

        if nan_flag:
            break

        # ______val
        model.eval()
        with torch.no_grad(): 
            vl, vr, vk, vs, r2_total = 0.0, 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            if args.kl_warmup_epochs > 0:
                val_beta = args.beta * min(1.0, epoch /args.kl_warmup_epochs)
            else:
                val_beta = args.beta

            Z_all = []
            for xb in val_loader:
                xb = xb.to(device)
                xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
                Z_all.append(z_traj.detach().cpu().numpy())
                
                # --------- Per-trial loss (MATLAB-style) ---------
                # xhat, xb: [B, L, N]

                # Compute reconstruction per trial (not averaged across batch)
                per_trial_rec = torch.mean((xhat - xb)**2, dim=(1,2))  # [B]

                # Compute the trial-wise KL term
                per_trial_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]

                # Compute the trial-wise smoothness term
                if args.lambda_smooth > 0:
                    per_trial_smooth = torch.mean(zdiff**2, dim=(1,2))  # [B]
                else:
                    per_trial_smooth = torch.zeros_like(per_trial_rec)

                # Combine into final per-trial loss vector
                per_trial_loss = per_trial_rec + val_beta * per_trial_kl + args.lambda_smooth * per_trial_smooth  # [B]

                # Accumulate properly
                vl += per_trial_loss.mean().item()
                vr += per_trial_rec.mean().item()
                vk += per_trial_kl.mean().item()
                vs += per_trial_smooth.mean().item()

                # xhat: [B, L, N_pca]
                Bv, Lv, Nv_pca = xhat.shape

                xhat_flat_pca = xhat.detach().cpu().numpy().reshape(-1, Nv_pca)  # [B*L, N_pca]
                xb_flat_pca   = xb.detach().cpu().numpy().reshape(-1, Nv_pca)    # [B*L, N_pca]

                xhat_raw_flat = pca.inverse_transform(xhat_flat_pca)  # [B*L, N_raw]
                xb_raw_flat   = pca.inverse_transform(xb_flat_pca)    # [B*L, N_raw]

                # R² for each trial in the batch
                for b in range(Bv):
                    xhat_raw_flat_b = pca.inverse_transform(xhat[b].detach().cpu().numpy())
                    xb_raw_flat_b   = pca.inverse_transform(xb[b].detach().cpu().numpy())

                    r2_b = compute_r2(xb_raw_flat_b, xhat_raw_flat_b)
                    if not np.isnan(r2_b):
                        r2_total += r2_b
                        n_batches += 1
                        
        z_flat = np.concatenate(Z_all, axis=0).reshape(-1, z_traj.shape[-1])  
        nbv = len(val_loader)
        mean_r2 = r2_total / max(1, n_batches)
        print(f"      valid loss {vl/nbv:.5f} | recon {vr/nbv:.5f} | kl {vk/nbv:.5f} | smooth {vs/nbv:.5f} | R² {mean_r2:.4f}")

        # quick preview image (first batch first trial)
        try:
                    xb = next(iter(val_loader)).to(device)
                    xhat, *_ = model(xb, tvec)
                    xb_np   = xb[0].detach().cpu().numpy()      # [L, N]
                    xhat_np = xhat[0].detach().cpu().numpy()
                    xb_raw = pca.inverse_transform(xb_np)
                    xhat_raw = pca.inverse_transform(xhat_np)
                    # plot mean across neurons for a quick sanity check
                    plt.figure(figsize=(8,3))
                    plt.plot(xb_raw.mean(axis=1), label="GT mean")
                    plt.plot(xhat_raw.mean(axis=1), label="Recon mean", alpha=0.8)
                    plt.legend(); plt.title("Validation mean activity (GT vs Recon)")
                    out_png = PATHS["preview"]
                    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
                    print(f"      wrote {out_png}")
        except Exception as e:
                    print("      (preview plot skipped:", e, ")")

    # --- Step 3: Latent manifold embedding (MIND-style) ---
    try:
        xb = next(iter(val_loader)).to(device)
        xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)

        # Landmark selection in LATENT SPACE (correct MIND method)
        k = getattr(args, "landmark_count", 300)

        lm_idx = greedy_landmarks(z_flat, k=k)

        # Pairwise distances only on landmarks
        D = pairwise_distances(z_flat[lm_idx], metric = "sqeuclidean")

        mds = MDS(n_components=3, dissimilarity='precomputed',
                random_state=0, n_init=1)
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
    
    # ---- FINAL SAVE ----
    ckpt = os.path.join(PATHS["out_dir"], "ode_vae_final.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "tvec": tvec_np,
        "meta": meta,
        "args": vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip() or "unknown",
        "data_hash": data_hash,
        "best_val_loss": best_val,
        "final_r2": mean_r2,
    }, ckpt)

    print(f"\nSaved FINAL model → {ckpt}\n")

    final_metrics = {
    "recon": vr / nbv,
    "kl": vk / nbv,
    "smooth": vs / nbv,
    "r2": mean_r2
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
    setup_run_logging(PATHS["run_output"])
    print(f"Running script: {os.path.basename(__file__)}")
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
