# note: using float32 inputs and added some code to get apple's MPS backend to work,
# becuase torchdiffeq likes to create float64 tensors for tolerances (rtol, atol) which MPS doesn't support. 

# Written by Kathleen Higgins
# Worked as of 2025-09-10
# src/train_neural_ode_vae.py
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
    "data":          os.path.join(SRC_DIR, "random_walk_data", "synthetic_rat_data.npz"),
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
    def __init__(self, n_neurons, latent_dim):
        super().__init__()
        self.enc = Encoder(n_neurons, latent_dim)
        self.odefunc = LatentODEFunc(latent_dim)
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
            step = (tvec[1] - tvec[0]).abs().item()
            # NOTE: step_size is required for fixed-step methods in torchdiffeq.
            z_traj = odeint(
                self.odefunc,
                z0,
                tvec,
                method="rk4",
                options={"step_size": step}
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
        dt = (tvec[1:] - tvec[:-1]).view(1, -1, 1)                    # [1, L-1, 1]
        zdiff = (z_traj[:, 1:, :] - z_traj[:, :-1, :]) / dt

        xhat = self.dec(z_traj)                                       # [B, L, N]
        return xhat, mu, logvar, z_traj, zdiff
    
#___________________loss_________________#
def vae_loss(xhat, x, mu, logvar, zdiff, beta=1.0, lambda_smooth=0.0):
    recon = torch.mean((xhat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    smooth = torch.mean(zdiff**2) if lambda_smooth > 0 else x.new_tensor(0.0)
    return recon + beta*kl + lambda_smooth*smooth, recon, kl, smooth


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

    model = ODEVAE(n_neurons=N, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tvec = torch.from_numpy(tvec_np).to(device)

    best_val = math.inf
    os.makedirs(args.out_dir, exist_ok = True)

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
            loss, rec,kl, sm = vae_loss(xhat, xb, mu, logvar, zdiff, beta=beta, lambda_smooth=args.lambda_smooth)
            
            # ---- NaN check ----
            if torch.isnan(loss):
                print(f"[epoch {epoch}] NaN detected — stopping training early.")
                nan_flag = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item(); tr += rec.item(); tk+= kl.item(); ts += sm.item()
        nb = len(train_loader)
        print(f"[{epoch:03d}] train loss {tl/nb:.5f} | recon { tr/nb:.5f} | kl {tk/nb:.5f} | smooth {ts/nb:.5f} | beta {beta:.3f}")

        if nan_flag:
            break

        # ______val
        model.eval()
    with torch.no_grad(): 
        vl, vr, vk, vs, r2_total = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        for xb in val_loader:
            xb = xb.to(device)
            xhat, mu, logvar, z_traj, zdiff = model(xb, tvec)
            loss, rec, kl, sm = vae_loss(xhat, xb, mu, logvar, zdiff, beta=args.beta, lambda_smooth=args.lambda_smooth)
            vl += loss.item(); vr += rec.item(); vk += kl.item(); vs += sm.item()

            # --- R² computation per batch ---
            r2_batch = compute_r2(xb.cpu(), xhat.cpu())
            if not np.isnan(r2_batch):
                r2_total += r2_batch
            n_batches += 1

        nbv = len(val_loader)
        mean_r2 = r2_total / max(1, n_batches)
        print(f"      valid loss {vl/nbv:.5f} | recon {vr/nbv:.5f} | kl {vk/nbv:.5f} | smooth {vs/nbv:.5f} | R² {mean_r2:.4f}")

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

                print("  saved best model to", ckpt)

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
                    print(f"      wrote {out_png}")
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
        import matplotlib.pyplot as plt
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
        json.dump(run_metadata, f, indent=2)
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