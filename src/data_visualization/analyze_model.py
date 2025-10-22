# analyze_model.py — One-shot visual diagnostics for ODE-VAE
# Runs after training. Saves all figures separately under pt_files/

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# 3D + animation + sklearn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from random import sample


# ---------- Path setup ----------
import os, sys

# This file now lives in: Neural_VAE_ODE/src/data_visualization/
VIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/data_visualization
SRC_DIR = os.path.dirname(VIS_DIR)                       # .../src
ROOT_DIR = os.path.dirname(SRC_DIR)                      # .../Neural_VAE_ODE

# Add src/ to Python path so we can import neural_ode_vae.py
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import from your training script
from neural_ode_vae import get_device, ODEVAE, make_sequences, SeqDataset

# ---------- Define path constants ----------
PATHS = {
    "data": os.path.join(SRC_DIR, "random_walk_data", "synthetic_rat_data.npz"),
    "checkpoint": os.path.join(SRC_DIR, "pt_files", "ode_vae_best.pt"),
    "out_dir": VIS_DIR,  # save all visualizations here
}

os.makedirs(PATHS["out_dir"], exist_ok=True)

device = get_device()
print(f"Using device: {device}")
print(f"Saving visualizations to: {PATHS['out_dir']}")

# --------- Load checkpoint (handle PyTorch 2.6 pickling change) ---------
ckpt_path = PATHS["checkpoint"]

# Allow safe numpy reconstruct used in older numpy pickles (only if you trust your own ckpt)
from torch.serialization import add_safe_globals
try:
    import numpy as _np
    add_safe_globals([_np._core.multiarray._reconstruct])  # allowlist the symbol
except Exception:
    pass

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# --------- Rebuild model and data ---------
args_loaded = checkpoint["args"]
meta = checkpoint["meta"]
tvec_np = checkpoint["tvec"]  # shape [L]
latent_dim = args_loaded["latent_dim"]

model = ODEVAE(n_neurons=meta["N"], latent_dim=latent_dim).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Load data & build trial sequences consistent with training
npz = np.load(PATHS["data"])
X, tvec_np, _ = make_sequences(
    npz,
    trial_len_s=args_loaded["trial_len_s"],
    fps=args_loaded["fps"],
    drop_first_trials=args_loaded["drop_first_trials"],
    min_frames=10
)
# Use last 10 sequences as a quick "validation-like" slice
valX = X[-10:] if X.shape[0] > 10 else X

# ---------- Helper for a single forward pass ----------
def run_one_trial(x_seq_np):
    xb = torch.from_numpy(x_seq_np[None, ...]).to(device)  # [1, L, N]
    with torch.no_grad():
        xhat, mu, logvar, z_traj, zdiff = model(xb, torch.from_numpy(tvec_np).to(device))
    return (xb[0].cpu().numpy(), xhat[0].cpu().numpy(),
            z_traj[0].cpu().numpy())  # [L, N], [L, N], [L, D]

# ======================================================== #
# ---------------- DATA VISUALIZATIONS ------------------- #
# ======================================================== #

# 1) Raw neural activity across session (random 5 neurons)
# --- Improved raw neural data visualization ---
roi_full = npz["roi"].T  # [T, N]
time_full = npz["Time"]
neurons = np.random.choice(roi_full.shape[1], size=min(5, roi_full.shape[1]), replace=False)

# Optional normalization for visualization (don’t distort relative differences)
roi_vis = (roi_full - np.min(roi_full, axis=0, keepdims=True)) / (np.ptp(roi_full, axis=0, keepdims=True) + 1e-8)

plt.figure(figsize=(10, 6))
offset = 2.0  # increase this if curves overlap
print(f"roi_full shape: {roi_full.shape}, time_full shape: {time_full.shape}")
print(f"Example values: roi_full mean={roi_full.mean():.3f}, std={roi_full.std():.3f}")

for i, n in enumerate(neurons):
    plt.plot(time_full, roi_vis[:, n] + i * offset, lw=1.5, label=f"Neuron {n}")
plt.title("Raw Neural Activity Across Time")
plt.xlabel("Time (s)")
plt.ylabel("Neuron (offset)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "raw_neural_activity.png"), dpi=160)
plt.close()



# Pick one trial to analyze in depth (first of valX for determinism)
xb_np, xhat_np, z_traj_np = run_one_trial(valX[0])

# 2) Neuron-level reconstructions (5 random neurons)
idxs = sample(range(xb_np.shape[1]), min(5, xb_np.shape[1]))
plt.figure(figsize=(10, 8))
for i, n in enumerate(idxs):
    plt.subplot(len(idxs), 1, i + 1)
    plt.plot(xb_np[:, n], label=f"GT neuron {n}", color='tab:blue', lw=1.5)
    plt.plot(xhat_np[:, n], label=f"Recon neuron {n}", color='tab:orange', lw=1.2, alpha=0.8)
    plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "neuron_recon_examples.png"), dpi=160)
plt.close()

# ======================================================== #
# --------------- LATENT DIAGNOSTICS --------------------- #
# ======================================================== #

# 3) Latent trajectories (first 8 dims)
plt.figure(figsize=(12, 6))
for d in range(min(8, z_traj_np.shape[1])):
    plt.plot(tvec_np, z_traj_np[:, d], label=f"z{d}", lw=1.5)
plt.title("Latent Trajectories (first 8 dims)")
plt.xlabel("Time (s)"); plt.ylabel("Latent value")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "latent_trajectories.png"), dpi=160)
plt.close()

# 4) 3D PCA of latent trajectory
pca3 = PCA(n_components=3)
z_pca3 = pca3.fit_transform(z_traj_np)  # [L, 3]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(z_pca3[:, 0], z_pca3[:, 1], z_pca3[:, 2], color="tab:blue", lw=2)
ax.set_title("3D Latent Trajectory (PCA projection)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "latent_pca_3d.png"), dpi=160)
plt.close()

# 5) Latent trajectory animation (2D PCA)
pca2 = PCA(n_components=2)
z_pca2 = pca2.fit_transform(z_traj_np)  # [L, 2]
fig, ax = plt.subplots(figsize=(6, 6))
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(z_pca2[:, 0].min() - 0.1, z_pca2[:, 0].max() + 0.1)
ax.set_ylim(z_pca2[:, 1].min() - 0.1, z_pca2[:, 1].max() + 0.1)
ax.set_title("Latent Trajectory (2D PCA)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

def _update(f):
    line.set_data(z_pca2[:f, 0], z_pca2[:f, 1])
    return (line,)

ani = animation.FuncAnimation(fig, _update, frames=len(z_pca2), blit=True, interval=40)
ani.save(os.path.join(PATHS["out_dir"], "latent_traj_anim.gif"), fps=25)
plt.close()

# 6) Latent vector field (2D)
# If latent_dim > 2, build a 2D projection with PCA on the *same* z_traj,
# and visualize vector field in that PCA space by pushing points back through
# the inverse linear map (approximate).
def _latent_field_2d(ode_func, proj, title, fname):
    # grid in 2D PCA space
    gx, gy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    grid2 = np.stack([gx, gy], axis=-1).reshape(-1, 2)  # [M, 2]

    # Map 2D -> D latent (inverse PCA: xD = grid2 @ components_ + mean_)
    gridD = grid2 @ proj.components_[:2, :] + proj.mean_
    gridD_t = torch.tensor(gridD, dtype=torch.float32, device=device)

    with torch.no_grad():
        dz = model.odefunc(torch.tensor(0.0, device=device), gridD_t).cpu().numpy()

    # Project dz to 2D: dz2 = dz @ components_[:2, :].T
    dz2 = dz @ proj.components_[:2, :].T

    plt.figure(figsize=(6, 6))
    plt.quiver(grid2[:, 0], grid2[:, 1], dz2[:, 0], dz2[:, 1], angles="xy", scale=30)
    plt.title(title)
    plt.xlabel("proj-1"); plt.ylabel("proj-2")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["out_dir"], fname), dpi=160)
    plt.close()

_latent_field_2d(model.odefunc, pca2, "Latent Vector Field (2D PCA space)", "latent_vector_field.png")

# 7) t-SNE of latent trajectory (color by time)
try:
    z_flat = z_traj_np  # [L, D]
    z_tsne = TSNE(n_components=2, perplexity=min(30, len(z_flat)//3 or 5), init="random").fit_transform(z_flat)
    plt.figure(figsize=(6,6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1],
                c=np.linspace(0, 1, len(z_tsne)), cmap="viridis", s=8)
    plt.title("t-SNE of Latent Trajectory (color=time)")
    plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["out_dir"], "latent_tsne.png"), dpi=160)
    plt.close()
except Exception as e:
    print("t-SNE skipped:", e)

# ======================================================== #
# ------------- RECONSTRUCTION DIAGNOSTICS --------------- #
# ======================================================== #

# 8) Per-neuron correlation (GT vs Recon) on the analyzed trial
corrs = []
for i in range(xb_np.shape[1]):
    xi = xb_np[:, i]
    yi = xhat_np[:, i]
    if np.allclose(xi.std(), 0) or np.allclose(yi.std(), 0):
        corrs.append(0.0)
    else:
        corrs.append(np.corrcoef(xi, yi)[0, 1])
plt.figure(figsize=(10, 3))
plt.plot(corrs, lw=1)
plt.title("Per-Neuron Reconstruction Correlation")
plt.xlabel("Neuron index"); plt.ylabel("r (GT vs Recon)")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_corr_per_neuron.png"), dpi=160)
plt.close()

# 9) Per-trial MSE (last 10 sequences)
from torch.utils.data import DataLoader
val_loader = DataLoader(SeqDataset(valX), batch_size=1, shuffle=False)
trial_errors = []
with torch.no_grad():
    for xb in val_loader:
        xb = xb.to(device)
        xhat, *_ = model(xb, torch.from_numpy(tvec_np).to(device))
        mse = torch.mean((xhat - xb) ** 2).item()
        trial_errors.append(mse)

plt.figure(figsize=(8, 4))
plt.plot(trial_errors, marker='o')
plt.title("Per-Trial Reconstruction Error (MSE)")
plt.xlabel("Validation trial #")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_mse_per_trial.png"), dpi=160)
plt.close()

# 10) Per-neuron MSE
mse_per_neuron = ((xhat_np - xb_np) ** 2).mean(axis=0)
plt.figure(figsize=(10, 3))
plt.plot(mse_per_neuron, color='tab:red')
plt.title("Per-Neuron Reconstruction MSE")
plt.xlabel("Neuron index"); plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_mse_per_neuron.png"), dpi=160)
plt.close()

# 11) Error heatmap (neurons × time)
err = (xhat_np - xb_np)
absmax = np.abs(err).max() or 1.0
plt.figure(figsize=(10, 6))
plt.imshow(err.T, aspect="auto", cmap="RdBu_r", vmin=-absmax, vmax=absmax)
plt.colorbar(label="Reconstruction Error (Recon − GT)")
plt.title("Neuron × Time Reconstruction Error Heatmap")
plt.xlabel("Time (frames)"); plt.ylabel("Neuron index")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_error_heatmap.png"), dpi=160)
plt.close()

print("Visualization complete. Files saved in:", PATHS["out_dir"])


# ======================================================== #
# ------- PER-NEURON MSE + RECONSTRUCTION HEATMAP -------- #
# ======================================================== #

# Compute reconstruction error
err = xhat_np - xb_np  # shape [T, N]
mse_per_neuron = np.mean(err ** 2, axis=0)
absmax = np.abs(err).max() or 1.0

# ----- Per-neuron reconstruction MSE -----
plt.figure(figsize=(10, 3))
plt.plot(mse_per_neuron, color='tab:red', lw=1.2)
plt.title("Per-Neuron Reconstruction MSE")
plt.xlabel("Neuron index")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_mse_per_neuron.png"), dpi=160)
plt.close()

print(f"Mean total MSE: {np.mean(mse_per_neuron):.6f}, Max per-neuron MSE: {np.max(mse_per_neuron):.6f}")

# ----- Neuron × Time error heatmap -----
plt.figure(figsize=(12, 6))
plt.imshow(
    err.T, 
    aspect="auto", 
    cmap="coolwarm", 
    vmin=-absmax, 
    vmax=absmax,
    interpolation="nearest"
)
plt.colorbar(label="Reconstruction Error (Recon − GT)")
plt.title("Neuron × Time Reconstruction Error Heatmap")
plt.xlabel("Time (frames)")
plt.ylabel("Neuron index")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "recon_error_heatmap.png"), dpi=160)
plt.close()

# ----- Mean absolute error summaries -----
mean_err_per_neuron = np.mean(np.abs(err), axis=0)
mean_err_per_time = np.mean(np.abs(err), axis=1)

plt.figure(figsize=(10, 3))
plt.plot(mean_err_per_neuron, color='tab:orange', lw=1.3)
plt.title("Mean Absolute Reconstruction Error per Neuron")
plt.xlabel("Neuron index")
plt.ylabel("Mean |Error|")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "mean_abs_error_per_neuron.png"), dpi=160)
plt.close()

plt.figure(figsize=(10, 3))
plt.plot(mean_err_per_time, color='tab:blue', lw=1.3)
plt.title("Mean Absolute Reconstruction Error over Time")
plt.xlabel("Time (frames)")
plt.ylabel("Mean |Error|")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "mean_abs_error_over_time.png"), dpi=160)
plt.close()

print("Saved per-neuron MSE and error heatmaps to:", PATHS["out_dir"])

# Raw versus Reconstruction Heatmap
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(xb_np.T, aspect='auto', cmap='viridis')
plt.title("Ground Truth Activity")
plt.xlabel("Time"); plt.ylabel("Neuron")

plt.subplot(1,2,2)
plt.imshow(xhat_np.T, aspect='auto', cmap='viridis')
plt.title("Reconstructed Activity")
plt.xlabel("Time"); plt.ylabel("Neuron")

plt.tight_layout()
plt.savefig(os.path.join(PATHS["out_dir"], "raw_vs_recon_heatmap.png"), dpi=160)
plt.close()

# Worst Neurons (Highlight Top Error Neurons):
worst_neurons = np.argsort(mse_per_neuron)[-5:]  # top 5 worst
for n in worst_neurons:
    plt.figure()
    plt.plot(xb_np[:, n], label=f"GT neuron {n}")
    plt.plot(xhat_np[:, n], label=f"Recon neuron {n}")
    plt.legend(); plt.title(f"High-Error Neuron {n}")
    plt.savefig(os.path.join(PATHS["out_dir"], f"worst_neuron_{n}.png"), dpi=150)
    plt.close()