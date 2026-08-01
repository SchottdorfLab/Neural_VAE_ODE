#!/usr/bin/env python3
"""Multi-trial version of simulate_geodesic_sphere.py.

This intentionally stays close to the original sphere sandbox script:
- same spherical_geodesic equation
- same Fibonacci tiling of place-field centers
- same von-Mises-like place-field activity
- same learned MetricNetwork / GeodesicDynamics / NeuralDecoder structure
- same free-dynamics comparison

The main change is that we simulate many sphere trajectories, each with a
different initial condition, and fit all of them jointly with one shared
geometry/decoder and per-trial x0/v0.
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim


def spherical_geodesic(t, y):
    """
    Computes the derivatives for the geodesic equations on a sphere.
    y = [theta, phi, dtheta/dt, dphi/dt]
    """
    theta, phi, dtheta, dphi = y

    epsilon = 1e-8  # Avoid exact division by zero at the poles

    # Geodesic equations derived from Christoffel symbols:
    # d^2(theta)/dt^2 = sin(theta) * cos(theta) * (dphi/dt)^2
    # d^2(phi)/dt^2   = -2 * cot(theta) * (dtheta/dt) * (dphi/dt)
    ddtheta = np.sin(theta) * np.cos(theta) * dphi**2
    ddphi = -2.0 * (np.cos(theta) / (np.sin(theta) + epsilon)) * dtheta * dphi

    return [dtheta, dphi, ddtheta, ddphi]


def random_initial_condition(speed=0.7):
    """Different starting point and tangent direction for each simulated trial."""
    theta = random.uniform(0.25 * np.pi, 0.75 * np.pi)
    phi = random.uniform(0.0, 2.0 * np.pi)
    direction = random.uniform(0.0, 2.0 * np.pi)
    dtheta = speed * np.cos(direction)
    dphi = speed * np.sin(direction) / max(np.sin(theta), 1e-3)
    return [theta, phi, dtheta, dphi]


def simulate_one_trial(y0, t_eval, theta_centers, phi_centers, kappa):
    # Solve the ODE
    sol = solve_ivp(spherical_geodesic, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method="RK45")
    theta_t = sol.y[0]
    phi_t = sol.y[1]

    # Calculate Neural Activity for all neurons over time
    N_neurons = len(theta_centers)
    activity = np.zeros((N_neurons, len(t_eval)))
    for i in range(N_neurons):
        # von-Mises tuning: the dot product of the unit vectors pointing to the neuron's center and the trajectory's current position.
        # I chose this ad hoc. Shouldn't really matter I think.
        product = (
            np.cos(theta_centers[i]) * np.cos(theta_t)
            + np.sin(theta_centers[i]) * np.sin(theta_t) * np.cos(phi_t - phi_centers[i])
        )
        activity[i, :] = np.exp(kappa * product)
    return activity, theta_t, phi_t


# Parameters are intentionally named like the original script where possible.
random.seed(int(os.environ.get("SPHERE_SEED", "42")))
np.random.seed(int(os.environ.get("SPHERE_SEED", "42")))
torch.manual_seed(int(os.environ.get("SPHERE_SEED", "42")))

device_name = os.environ.get("GEODESIC_DEVICE")
if device_name:
    device = torch.device(device_name)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

num_trials = int(os.environ.get("SPHERE_N_TRIALS", "24"))
N_neurons = int(os.environ.get("SPHERE_N_NEURONS", "96"))
kappa = float(os.environ.get("SPHERE_KAPPA", "1.5"))  # Tuning Width
speed = float(os.environ.get("SPHERE_SPEED", "0.7"))
t_span = (0, float(os.environ.get("SPHERE_T_MAX", str(4 * np.pi))))
t_eval = np.linspace(t_span[0], t_span[1], int(os.environ.get("SPHERE_N_TIME", "80")))
out_dir = Path(os.environ.get("SPHERE_OUT_DIR", "runs/geodesic_sphere_trials")).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
model_solver = os.environ.get("SPHERE_MODEL_SOLVER", "rk4").strip().lower()
if model_solver not in {"rk4", "euler"}:
    raise ValueError(f"Unknown SPHERE_MODEL_SOLVER={model_solver!r}; use 'rk4' or 'euler'.")

# Tile the sphere with "place field"
indices = np.arange(0, N_neurons, dtype=float) + 0.5
phi_centers = np.pi * (1 + 5**0.5) * indices
theta_centers = np.arccos(1 - 2 * indices / N_neurons)

# Simulate many trials, all sharing the same place fields.
dataset_train = []
activities = []
true_latents = []
initial_conditions = []
for trial_idx in range(num_trials):
    y0 = random_initial_condition(speed=speed)
    activity, theta_t, phi_t = simulate_one_trial(y0, t_eval, theta_centers, phi_centers, kappa)
    dataset_train.append({
        "idx": trial_idx,
        "rates": torch.tensor(activity.T, dtype=torch.float32, device=device),
        "seq_len": len(t_eval),
    })
    activities.append(activity.T.astype(np.float32))
    true_latents.append(np.stack([theta_t, phi_t], axis=1).astype(np.float32))
    initial_conditions.append(np.asarray(y0, dtype=np.float32))

activities = np.stack(activities, axis=0)
true_latents = np.stack(true_latents, axis=0)
initial_conditions = np.stack(initial_conditions, axis=0)

print(f"Simulated {num_trials} sphere trials: time={len(t_eval)}, neurons={N_neurons}")
print(f"Model solver: {model_solver}")
np.savez_compressed(
    out_dir / "synthetic_sphere_trials.npz",
    activities=activities,
    true_latents=true_latents,
    initial_conditions=initial_conditions,
    theta_centers=theta_centers.astype(np.float32),
    phi_centers=phi_centers.astype(np.float32),
    t_eval=t_eval.astype(np.float32),
)

if os.environ.get("SPHERE_GENERATE_ONLY", "").lower() in {"1", "true", "yes"}:
    print(f"Saved synthetic sphere trials to {out_dir / 'synthetic_sphere_trials.npz'}")
    raise SystemExit(0)


# =============== Geodesic fit ===================#

class MetricNetwork(nn.Module):
    """
    Parametrizes the metric tensor as a neural net.
    """
    def __init__(self, latent_dim=2, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        # MLP to predict the elements of the Cholesky factor L given coordinates.
        # For a d-dimensional space, L has d(d+1)/2 non-zero elements
        out_dim = latent_dim * (latent_dim + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.eps = 1e-4  # Minimum bound to prevent singular matrices

    def forward(self, x):
        """
        Maps latent coordinate x to a positive-definite metric tensor g(x).
        x shape: [batch_size, latent_dim]
        """
        batch_size = x.shape[0]
        out = self.net(x)

        # Construct lower triangular matrix L
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=x.device)

        # Fill the Cholesky factor row by row. For d=2 this preserves the
        # original ordering: L11, L21, L22.
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                value = torch.exp(out[:, idx]) if i == j else out[:, idx]
                L[:, i, j] = value
                idx += 1

        # g = L * L^T + eps * I
        I = torch.eye(self.latent_dim, device=x.device).unsqueeze(0)
        g = torch.bmm(L, L.transpose(1, 2)) + self.eps * I
        return g


class GeodesicDynamics(nn.Module):
    def __init__(self, metric_net):
        super().__init__()
        self.metric_net = metric_net

    def compute_christoffel(self, x):
        """
        Computes Christoffel as derivatives of metic tensory, using PyTorch Autograd.
        """
        x.requires_grad_(True)
        g = self.metric_net(x)  # [batch, d, d]
        batch_size, d, _ = g.shape

        # Step 1. Compute spatial derivatives of the metric tensor (dg_ij / dx_k)
        dg = torch.zeros(batch_size, d, d, d, device=x.device)
        for i in range(d):
            for j in range(d):
                # Gradients of g_{ij} with respect to all x
                grad_g = torch.autograd.grad(
                    outputs=g[:, i, j].sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True
                )[0]
                dg[:, i, j, :] = grad_g  # Shape: [batch, d]

        # 2. Compute inverse metric g^{kl}
        g_inv = torch.inverse(g)

        # 3. Construct Christoffel symbols
        Gamma = torch.zeros(batch_size, d, d, d, device=x.device)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    term = 0
                    for l in range(d):
                        term += 0.5 * g_inv[:, k, l] * (
                            dg[:, n, l, m] + dg[:, m, l, n] - dg[:, m, n, l]
                        )
                    Gamma[:, k, m, n] = term
        return Gamma

    def acceleration(self, x, v):
        Gamma = self.compute_christoffel(x)
        d = x.shape[1]

        # Compute acceleration: a^k = - \sum_{m,n} Gamma^k_{mn} v^m v^n
        a = torch.zeros_like(v)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    a[:, k] -= Gamma[:, k, m, n] * v[:, m] * v[:, n]
        return a

    def _euler_step(self, x, v, dt):
        a = self.acceleration(x, v)

        v_next = v + a * dt
        x_next = x + v * dt
        return x_next, v_next

    def _rk4_step(self, x, v, dt):
        k1_v = self.acceleration(x, v)
        k1_x = v

        k2_v = self.acceleration(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
        k2_x = v + 0.5 * dt * k1_v

        k3_v = self.acceleration(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
        k3_x = v + 0.5 * dt * k2_v

        k4_v = self.acceleration(x + dt * k3_x, v + dt * k3_v)
        k4_x = v + dt * k3_v

        v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        return x_next, v_next

    def forward(self, state, dt, solver="rk4"):
        """
        Performs one integration step of the second-order geodesic ODE.
        state: [x, v] where x and v are [batch, latent_dim]
        """
        x, v = state
        if solver == "euler":
            return self._euler_step(x, v, dt)
        return self._rk4_step(x, v, dt)


class NeuralDecoder(nn.Module):
    """
    MLP from latent space to neural activity.
    """
    def __init__(self, latent_dim=2, n_neurons=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_neurons),
            nn.Softplus()  # Firing rates must be strictly positive
        )

    def forward(self, x):
        return self.net(x)


class InverseGeodesicModel(nn.Module):
    """
    Assembles the model:
    1. Metric network is the NN model for gij
    2. Geodesic dynamics are the dynamics produced from (1)
    3. Neural decoder produces firing rates.
    -> Gradient-descent the whole thing
    """
    def __init__(self, num_trials, latent_dim=2, n_neurons=300, solver="rk4"):
        super().__init__()
        self.metric = MetricNetwork(latent_dim)
        self.dynamics = GeodesicDynamics(self.metric)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        self.solver = solver

        # Treat the initial states as learnable parameters, one per trial.
        self.x0 = nn.Parameter(torch.randn(num_trials, latent_dim))
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim))

    def forward(self, trial_idx, t_eval):
        """
        Rolls out the latent trajectory and decodes to neural activity.
        """
        dt = t_eval[1] - t_eval[0]  # Assuming uniform time steps
        seq_len = len(t_eval)

        single_trial = not torch.is_tensor(trial_idx) or trial_idx.ndim == 0
        if not torch.is_tensor(trial_idx):
            trial_idx = torch.tensor([trial_idx], dtype=torch.long, device=self.x0.device)
        elif trial_idx.ndim == 0:
            trial_idx = trial_idx.reshape(1).to(device=self.x0.device, dtype=torch.long)
        else:
            trial_idx = trial_idx.to(device=self.x0.device, dtype=torch.long)

        x = self.x0[trial_idx]
        v = self.v0[trial_idx]
        latents = []

        # Roll out all requested trials together. Trials remain independent; the
        # leading tensor dimension is only a compute batch.
        for _ in range(seq_len):
            latents.append(x)
            x, v = self.dynamics((x, v), dt, solver=self.solver)

        latents = torch.stack(latents, dim=1)  # [batch, seq_len, latent_dim]

        # Decode to neural firing rates
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_rates = self.decoder(flat_latents)
        rates = flat_rates.reshape(latents.shape[0], seq_len, -1)  # [batch, seq_len, n_neurons]
        if single_trial:
            return latents[0], rates[0]
        return latents, rates


# =============== Model comparison with a usual neural ODE ===================#

class FreeDynamics(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128):
        super().__init__()
        # Maps the concatenated state [x, v] directly to acceleration. No geodesics here.
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def acceleration(self, x, v):
        state_vec = torch.cat([x, v], dim=-1)  # Shape: [batch, latent_dim * 2]
        return self.net(state_vec)

    def _euler_step(self, x, v, dt):
        a = self.acceleration(x, v)
        v_next = v + a * dt
        x_next = x + v * dt
        return x_next, v_next

    def _rk4_step(self, x, v, dt):
        k1_v = self.acceleration(x, v)
        k1_x = v

        k2_v = self.acceleration(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
        k2_x = v + 0.5 * dt * k1_v

        k3_v = self.acceleration(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
        k3_x = v + 0.5 * dt * k2_v

        k4_v = self.acceleration(x + dt * k3_x, v + dt * k3_v)
        k4_x = v + dt * k3_v

        v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        return x_next, v_next

    def forward(self, state, dt, solver="rk4"):
        """
        Performs one integration step using unconstrained neural dynamics.
        """
        x, v = state
        if solver == "euler":
            return self._euler_step(x, v, dt)
        return self._rk4_step(x, v, dt)


class InverseFreeModel(nn.Module):
    def __init__(self, num_trials, latent_dim=2, n_neurons=300, solver="rk4"):
        super().__init__()
        self.dynamics = FreeDynamics(latent_dim)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        self.solver = solver

        # Learnable initial conditions, one per trial.
        self.x0 = nn.Parameter(torch.randn(num_trials, latent_dim))
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim))

    def forward(self, trial_idx, t_eval):
        dt = t_eval[1] - t_eval[0]
        seq_len = len(t_eval)

        single_trial = not torch.is_tensor(trial_idx) or trial_idx.ndim == 0
        if not torch.is_tensor(trial_idx):
            trial_idx = torch.tensor([trial_idx], dtype=torch.long, device=self.x0.device)
        elif trial_idx.ndim == 0:
            trial_idx = trial_idx.reshape(1).to(device=self.x0.device, dtype=torch.long)
        else:
            trial_idx = trial_idx.to(device=self.x0.device, dtype=torch.long)

        x = self.x0[trial_idx]
        v = self.v0[trial_idx]
        latents = []

        for _ in range(seq_len):
            latents.append(x)
            x, v = self.dynamics((x, v), dt, solver=self.solver)

        latents = torch.stack(latents, dim=1)
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_rates = self.decoder(flat_latents)
        rates = flat_rates.reshape(latents.shape[0], seq_len, -1)
        if single_trial:
            return latents[0], rates[0]
        return latents, rates


def train_and_evaluate(model, dataset, t_eval_np, epochs=300, lr=1e-3):
    t_eval_torch = torch.tensor(t_eval_np, dtype=torch.float32, device=device)
    trial_indices = torch.tensor([trial["idx"] for trial in dataset], dtype=torch.long, device=device)
    target_rates = torch.stack([trial["rates"] for trial in dataset], dim=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We need the SUM of the negative log-likelihood for exact AIC/BIC scaling. I messed this up
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred_latents, pred_rates = model(trial_indices, t_eval_torch)

        # The loss here is the negative log-likelihood (NLL)
        # (excluding the constant term, which drops out in model comparison)
        nll_sum = loss_fn(pred_rates, target_rates)

        nll_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_history.append(float(nll_sum.detach().cpu()))

        if epoch % int(os.environ.get("SPHERE_LOG_EVERY", "10")) == 0:
            print(f"Epoch {epoch:03d} | NLL Sum: {float(nll_sum.detach().cpu()):.2f}")

    return model, loss_history, float(nll_sum.detach().cpu())


def calculate_ic(nll_sum, num_params, num_obs):
    aic = 2 * num_params + 2 * nll_sum
    bic = num_params * np.log(num_obs) + 2 * nll_sum
    return aic, bic


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_all(model, dataset, t_eval_np):
    t_eval_torch = torch.tensor(t_eval_np, dtype=torch.float32, device=device)
    trial_indices = torch.tensor([trial["idx"] for trial in dataset], dtype=torch.long, device=device)
    pred_latents, pred_rates = model(trial_indices, t_eval_torch)
    return pred_rates.detach().cpu().numpy(), pred_latents.detach().cpu().numpy()


def corr_and_r2(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    r = np.corrcoef(yt, yp)[0, 1]
    r2 = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    return float(r), float(r2)


def plot_model_heatmap(true_rates, rates_geo_pred, rates_free_pred, dataset_idx=0, num_neurons=50):
    """Plots true vs predicted population rates as side-by-side heatmaps."""
    n_plot = min(num_neurons, true_rates.shape[2])
    mat_true = true_rates[dataset_idx, :, :n_plot].T
    mat_geo = rates_geo_pred[dataset_idx, :, :n_plot].T
    mat_free = rates_free_pred[dataset_idx, :, :n_plot].T

    vmin = min(mat_true.min(), mat_geo.min(), mat_free.min())
    vmax = max(mat_true.max(), mat_geo.max(), mat_free.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    im0 = axes[0].imshow(mat_true, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("True Data")
    axes[0].set_ylabel("Neuron Index")
    axes[0].set_xlabel("Time Step")

    im1 = axes[1].imshow(mat_geo, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("Geodesic Fit")
    axes[1].set_xlabel("Time Step")

    im2 = axes[2].imshow(mat_free, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title("Free ODE Fit")
    axes[2].set_xlabel("Time Step")

    fig.colorbar(im2, ax=axes.ravel().tolist(), label="Firing Rate", shrink=0.8)
    plt.suptitle(f"Population Activity Heatmaps for Simulated Trial {dataset_idx}", fontsize=14, y=1.02)
    plt.savefig(out_dir / "sphere_trial_reconstruction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_latents(true_z, geo_z, free_z):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, z, title in zip(axes, [true_z, geo_z, free_z], ["True theta/phi", "Geodesic latent", "Free latent"]):
        for trial_idx in range(min(z.shape[0], 12)):
            ax.plot(z[trial_idx, :, 0], z[trial_idx, :, 1], alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_dir / "sphere_latent_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Parameters
latent_dim = int(os.environ.get("SPHERE_LATENT_DIM", "2"))
n_timepoints = len(t_eval)
num_obs = num_trials * N_neurons * n_timepoints
epochs = int(os.environ.get("SPHERE_EPOCHS", "60"))
lr = float(os.environ.get("SPHERE_LR", "0.001"))

# Init models
model_geo = InverseGeodesicModel(num_trials=num_trials, latent_dim=latent_dim, n_neurons=N_neurons, solver=model_solver).to(device)
model_free = InverseFreeModel(num_trials=num_trials, latent_dim=latent_dim, n_neurons=N_neurons, solver=model_solver).to(device)

params_geo = count_parameters(model_geo)
params_free = count_parameters(model_free)
print(f"Geodesic Model Parameters: {params_geo}")
print(f"Free Dynamics Model Parameters: {params_free}\n")

# Train models
model_geo, loss_geo, nll_geo = train_and_evaluate(model_geo, dataset_train, t_eval, epochs=epochs, lr=lr)
print("-" * 20)
model_free, loss_free, nll_free = train_and_evaluate(model_free, dataset_train, t_eval, epochs=epochs, lr=lr)

rates_geo_pred, latents_geo = predict_all(model_geo, dataset_train, t_eval)
rates_free_pred, latents_free = predict_all(model_free, dataset_train, t_eval)
r_geo, r2_geo = corr_and_r2(activities, rates_geo_pred)
r_free, r2_free = corr_and_r2(activities, rates_free_pred)

# Get AIC/BIC
aic_geo, bic_geo = calculate_ic(nll_geo, params_geo, num_obs)
aic_free, bic_free = calculate_ic(nll_free, params_free, num_obs)

print("\n" + "=" * 45)
print("            MODEL COMPARISON RESULTS ")
print("=" * 45)
print(f"{'Metric':<15} | {'Geodesic Model':<15} | {'Free Model':<15}")
print("-" * 48)
print(f"{'Parameters (k)':<15} | {params_geo:<15} | {params_free:<15}")
print(f"{'Final NLL':<15} | {nll_geo:<15.2f} | {nll_free:<15.2f}")
print(f"{'R2':<15} | {r2_geo:<15.4f} | {r2_free:<15.4f}")
print(f"{'r':<15} | {r_geo:<15.4f} | {r_free:<15.4f}")
print(f"{'AIC':<15} | {aic_geo:<15.2f} | {aic_free:<15.2f}")
print(f"{'BIC':<15} | {bic_geo:<15.2f} | {bic_free:<15.2f}")
print("=" * 45)

best_aic = "Geodesic" if aic_geo < aic_free else "Free Dynamics"
best_bic = "Geodesic" if bic_geo < bic_free else "Free Dynamics"
print(f"\nPreferred Model by AIC: {best_aic}")
print(f"Preferred Model by BIC: {best_bic}")

plot_model_heatmap(activities, rates_geo_pred, rates_free_pred, dataset_idx=0, num_neurons=50)
plot_latents(true_latents, latents_geo, latents_free)

summary = {
    "config": {
        "num_trials": num_trials,
        "N_neurons": N_neurons,
        "n_timepoints": n_timepoints,
        "latent_dim": latent_dim,
        "kappa": kappa,
        "speed": speed,
        "epochs": epochs,
        "lr": lr,
        "device": str(device),
        "model_solver": model_solver,
    },
    "geodesic": {"params": params_geo, "final_nll": nll_geo, "R2": r2_geo, "r": r_geo, "AIC": aic_geo, "BIC": bic_geo},
    "free": {"params": params_free, "final_nll": nll_free, "R2": r2_free, "r": r_free, "AIC": aic_free, "BIC": bic_free},
    "preferred_by_AIC": best_aic,
    "preferred_by_BIC": best_bic,
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
np.savez_compressed(
    out_dir / "fit_outputs.npz",
    activities=activities.astype(np.float32),
    rates_geo_pred=rates_geo_pred.astype(np.float32),
    rates_free_pred=rates_free_pred.astype(np.float32),
    true_latents=true_latents.astype(np.float32),
    latents_geo=latents_geo.astype(np.float32),
    latents_free=latents_free.astype(np.float32),
    loss_geo=np.asarray(loss_geo, dtype=np.float32),
    loss_free=np.asarray(loss_free, dtype=np.float32),
)
print(f"Saved outputs to {out_dir}")
