#!/usr/bin/env python3
"""
Same things as the original simulate_geodesics.py script,
but supports multiple runs, plus
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

TORCH_DTYPE = torch.float64
NP_DTYPE = np.float64
torch.set_default_dtype(TORCH_DTYPE)

# for the reconstruction graphs 
TRUE_COLOR = "#12355B"
RECON_COLOR = "#FF5A1F"
SPHERE_COLOR = "#8E8E8E"
WIREFRAME_COLOR = "#5F5F5F"
TRUE_LINEWIDTH = 3.4
RECON_LINEWIDTH = 3.4

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


def random_initial_condition(speed=2**-0.5):
    """Different starting point and tangent direction for each simulated trial."""
    theta = random.uniform(0.25 * np.pi, 0.75 * np.pi)
    # theta = np.pi / 2 # to test only coordinates at the equator
    phi = random.uniform(0.0, 2.0 * np.pi)
    direction = random.uniform(0.0, 2.0 * np.pi)
    # dtheta = 0.0
    dtheta = speed * np.cos(direction)
    dphi = speed * np.sin(direction) / max(np.sin(theta), 1e-3)
    return [theta, phi, dtheta, dphi]


def initial_condition_for_trial(trial_idx, speed):
    return random_initial_condition(speed=speed)


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


def plot_generated_activity(activity, theta_t, phi_t, theta_centers, phi_centers, t_eval, out_path):
    fig = plt.figure(figsize=(14, 6))

    # Plot A: 3D Sphere, Trajectory, and Neuron Centers
    ax1 = fig.add_subplot(131, projection="3d")

    # Draw transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ax1.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones(np.size(u)), np.cos(v)),
        color="cyan",
        alpha=0.1,
        edgecolor="none",
    )

    # Convert trajectory to Cartesian
    x_t = np.sin(theta_t) * np.cos(phi_t)
    y_t = np.sin(theta_t) * np.sin(phi_t)
    z_t = np.cos(theta_t)
    ax1.plot(x_t, y_t, z_t, color="red", linewidth=3, label="Agent Trajectory")

    # Convert neuron centers to Cartesian
    x_c = np.sin(theta_centers) * np.cos(phi_centers)
    y_c = np.sin(theta_centers) * np.sin(phi_centers)
    z_c = np.cos(theta_centers)
    ax1.scatter(x_c, y_c, z_c, color="black", s=10, alpha=0.6, label="Place Cell Centers")

    ax1.set_title("Geodesic trajectory + place field centers")
    ax1.legend()

    ax2 = fig.add_subplot(132)  # Raster plot of Neural Activity
    peak_times = np.argmax(activity, axis=1)
    peak_idx = np.argsort(peak_times)
    random.shuffle(peak_idx)
    random_activity = activity[peak_idx, :]  # Random ordering of neurons

    im = ax2.imshow(random_activity, aspect="auto", origin="lower", cmap="magma", extent=[t_eval[0], t_eval[-1], 0, activity.shape[0]])
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Neuron ID (Sorted by Peak Activation)")
    ax2.set_title("Neural Population Activity")
    plt.colorbar(im, ax=ax2, label="Normalized Firing Rate")

    ax3 = fig.add_subplot(133)  # Sort neurons by their peak activity time to visualize the sequence
    peak_idx = np.argsort(peak_times)
    sorted_activity = activity[peak_idx, :]  # Order neurons by peak time. Shows the sequential activation of the place fields
    im = ax3.imshow(sorted_activity, aspect="auto", origin="lower", cmap="magma", extent=[t_eval[0], t_eval[-1], 0, activity.shape[0]])
    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("Neuron ID (Sorted by Peak Activation)")
    ax3.set_title("Neural Population Activity (sorted))")
    plt.colorbar(im, ax=ax3, label="Normalized Firing Rate")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


seed = 42 # I'm just manually setting this here 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device_name = os.environ.get("DEVICE")
if device_name:
    device = torch.device(device_name)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if device.type == "mps":
    raise RuntimeError(
        "fit_sphere_trials.py uses float64, which PyTorch MPS does not support. "
        "Set DEVICE=cpu locally or DEVICE=cuda on a CUDA system."
    )
print(f"Using device: {device}")
print(f"Using dtype: {TORCH_DTYPE}")

# The defaults here are the same as simulate_geodesic_sphere.py, they just make multi-trial and 3d tests possible 
# without changing the original constants.
num_trials = int(os.environ.get("SPHERE_N_TRIALS", "1"))
N_neurons = int(os.environ.get("SPHERE_N_NEURONS", "300"))
kappa = 1.5  # Tuning Width
speed = float(os.environ.get("SPHERE_SPEED", str(2**-0.5)))
t_span = (0, float(os.environ.get("SPHERE_T_MAX", str(4 * np.pi))))  # Integrate long enough to wrap around the sphere
t_eval = np.linspace(t_span[0], t_span[1], int(os.environ.get("SPHERE_N_TIME", "600")))
out_dir = Path(os.environ.get("SPHERE_OUT_DIR", "runs/geodesic_sphere_trials")).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
model_solver = os.environ.get("SPHERE_MODEL_SOLVER", "euler").strip().lower()
if model_solver not in {"rk4", "euler"}:
    raise ValueError(f"Unknown SPHERE_MODEL_SOLVER={model_solver!r}; use 'rk4' or 'euler' please!")

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
    y0 = initial_condition_for_trial(trial_idx, speed=speed)
    activity, theta_t, phi_t = simulate_one_trial(y0, t_eval, theta_centers, phi_centers, kappa)
    dataset_train.append({
        "idx": trial_idx,
        "rates": torch.tensor(activity.T, dtype=TORCH_DTYPE, device=device),
        "seq_len": len(t_eval),
    })
    activities.append(activity.T.astype(NP_DTYPE))
    true_latents.append(np.stack([theta_t, phi_t], axis=1).astype(NP_DTYPE))
    initial_conditions.append(np.asarray(y0, dtype=NP_DTYPE))
    if trial_idx == 0:
        plot_generated_activity(
            activity,
            theta_t,
            phi_t,
            theta_centers,
            phi_centers,
            t_eval,
            out_dir / "sphere_generated_activity.png",
        )
heldout_frac = float(os.environ.get("SPHERE_HELDOUT_FRAC", "0.2"))
if not 0.0 <= heldout_frac < 1.0:
    raise ValueError("SPHERE_HELDOUT_FRAC must be in [0, 1).")

if len(dataset_train) > 1 and heldout_frac > 0.0:
    train_size = int((1.0 - heldout_frac) * len(dataset_train))
    train_size = max(1, train_size)
    heldout_size = len(dataset_train) - train_size
else:
    train_size = len(dataset_train)
    heldout_size = 0

if heldout_size > 0:
    train_dataset, heldout_dataset = random_split(dataset_train, [train_size, heldout_size])
else:
    train_dataset, heldout_dataset = dataset_train, []
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
heldout_loader = DataLoader(heldout_dataset, batch_size = 32, shuffle = False)

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
    theta_centers=theta_centers.astype(NP_DTYPE),
    phi_centers=phi_centers.astype(NP_DTYPE),
    t_eval=t_eval.astype(NP_DTYPE),
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
        L = torch.zeros(
            batch_size,
            self.latent_dim,
            self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )

        # Fill the Cholesky factor row by row. For d=2 this preserves the
        # original ordering: L11, L21, L22.
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                value = torch.exp(out[:, idx]) if i == j else out[:, idx]
                L[:, i, j] = value
                idx += 1

        # g = L * L^T + eps * I
        I = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
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
        dg = torch.zeros(batch_size, d, d, d, device=x.device, dtype=x.dtype)
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
        Gamma = torch.zeros(batch_size, d, d, d, device=x.device, dtype=x.dtype)
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
    t_eval_torch = torch.tensor(t_eval_np, dtype=TORCH_DTYPE, device=device)
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


def fit_heldout_initial_conditions(model, dataset, t_eval_np, epochs=300, lr=1e-3):
    """Fit only heldout trial initial conditions while shared dynamics/decoder stay fixed."""
    if len(dataset) == 0:
        return None

    t_eval_torch = torch.tensor(t_eval_np, dtype=TORCH_DTYPE, device=device)
    trial_indices = torch.tensor([trial["idx"] for trial in dataset], dtype=torch.long, device=device)
    target_rates = torch.stack([trial["rates"] for trial in dataset], dim=0)
    previous_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    for name, p in model.named_parameters():
        p.requires_grad_(name in {"x0", "v0"})

    optimizer = optim.Adam([model.x0, model.v0], lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    loss_history = []

    try:
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, pred_rates = model(trial_indices, t_eval_torch)
            nll_sum = loss_fn(pred_rates, target_rates)
            nll_sum.backward()
            torch.nn.utils.clip_grad_norm_([model.x0, model.v0], max_norm=1.0)
            optimizer.step()
            loss_history.append(float(nll_sum.detach().cpu()))

        _, pred_rates = model(trial_indices, t_eval_torch)
        nll_sum = loss_fn(pred_rates, target_rates)
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(previous_requires_grad[name])

    return float(nll_sum.detach().cpu()), loss_history


def calculate_ic(nll_sum, num_params, num_obs):
    aic = 2 * num_params + 2 * nll_sum
    bic = num_params * np.log(num_obs) + 2 * nll_sum
    return aic, bic


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_all(model, dataset, t_eval_np):
    t_eval_torch = torch.tensor(t_eval_np, dtype=TORCH_DTYPE, device=device)
    trial_indices = torch.tensor([trial["idx"] for trial in dataset], dtype=torch.long, device=device)
    pred_latents, pred_rates = model(trial_indices, t_eval_torch)
    return pred_rates.detach().cpu().numpy(), pred_latents.detach().cpu().numpy()


def rates_numpy(dataset):
    if len(dataset) == 0:
        return np.empty((0, len(t_eval), N_neurons), dtype=NP_DTYPE)
    return torch.stack([trial["rates"] for trial in dataset], dim=0).detach().cpu().numpy()


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


def theta_phi_to_xyz(theta_phi):
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )


def centers_to_xyz(theta_centers_np, phi_centers_np):
    return np.stack(
        [
            np.sin(theta_centers_np) * np.cos(phi_centers_np),
            np.sin(theta_centers_np) * np.sin(phi_centers_np),
            np.cos(theta_centers_np),
        ],
        axis=1,
    )


def rates_to_sphere_xyz(rates, theta_centers_np, phi_centers_np):
    """Decode neural activity to a point on the unit sphere by population vector."""
    centers_xyz = centers_to_xyz(theta_centers_np, phi_centers_np)
    flat_rates = np.clip(rates.reshape(-1, rates.shape[-1]).astype(np.float64), 0.0, None)
    xyz = flat_rates @ centers_xyz
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / np.maximum(norm, 1e-12)
    return xyz.reshape(*rates.shape[:-1], 3)


def draw_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 96)
    v = np.linspace(0, np.pi, 48)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=SPHERE_COLOR, alpha=0.26, linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color=WIREFRAME_COLOR, alpha=0.12, linewidth=0.45, rstride=5, cstride=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=24, azim=38)


def draw_prediction_paths(ax, true_theta_phi, pred_rates, theta_centers_np, phi_centers_np, max_trials):
    true_xyz = theta_phi_to_xyz(true_theta_phi)
    pred_xyz = rates_to_sphere_xyz(pred_rates, theta_centers_np, phi_centers_np)
    n_trials = min(true_xyz.shape[0], pred_xyz.shape[0], max_trials)
    for trial_idx in range(n_trials):
        n_time = min(true_xyz[trial_idx].shape[0], pred_xyz[trial_idx].shape[0])
        step = max(1, n_time // 260)
        ax.plot(
            true_xyz[trial_idx, :n_time:step, 0],
            true_xyz[trial_idx, :n_time:step, 1],
            true_xyz[trial_idx, :n_time:step, 2],
            color=TRUE_COLOR,
            lw=TRUE_LINEWIDTH,
            alpha=0.88,
        )
        ax.plot(
            pred_xyz[trial_idx, :n_time:step, 0],
            pred_xyz[trial_idx, :n_time:step, 1],
            pred_xyz[trial_idx, :n_time:step, 2],
            color=RECON_COLOR,
            lw=RECON_LINEWIDTH,
            alpha=0.98,
        )


def plot_sphere_prediction_overlay(true_theta_phi, pred_rates, theta_centers_np, phi_centers_np, out_path, title, max_trials=12):
    if true_theta_phi.shape[0] == 0 or pred_rates.shape[0] == 0:
        return
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_sphere(ax)
    draw_prediction_paths(ax, true_theta_phi, pred_rates, theta_centers_np, phi_centers_np, max_trials)
    ax.set_title(title, pad=18)
    ax.legend(
        handles=[
            Line2D([0], [0], color=TRUE_COLOR, lw=TRUE_LINEWIDTH, label="true path"),
            Line2D([0], [0], color=RECON_COLOR, lw=RECON_LINEWIDTH, label="predicted path"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sphere_prediction_comparison(
    true_theta_phi,
    rates_geo_pred_np,
    rates_free_pred_np,
    theta_centers_np,
    phi_centers_np,
    out_path,
    split_label,
    max_trials=12,
):
    if true_theta_phi.shape[0] == 0:
        return
    fig = plt.figure(figsize=(15, 7.5))
    panels = [
        ("Geodesic prediction", rates_geo_pred_np),
        ("Free prediction", rates_free_pred_np),
    ]
    for panel_idx, (title, pred_rates) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        draw_sphere(ax)
        draw_prediction_paths(ax, true_theta_phi, pred_rates, theta_centers_np, phi_centers_np, max_trials)
        ax.set_title(f"{title} ({split_label})", pad=16)
    fig.legend(
        handles=[
            Line2D([0], [0], color=TRUE_COLOR, lw=TRUE_LINEWIDTH, label="true path"),
            Line2D([0], [0], color=RECON_COLOR, lw=RECON_LINEWIDTH, label="predicted path"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Parameters
latent_dim = int(os.environ.get("SPHERE_LATENT_DIM", "2"))
n_timepoints = len(t_eval)
num_obs = len(train_dataset) * N_neurons * n_timepoints
epochs = int(os.environ.get("SPHERE_EPOCHS", os.environ.get("GEODESIC_COMPARE_EPOCHS", "300")))
lr = float(os.environ.get("SPHERE_LR", "0.001"))
heldout_fit_epochs = int(os.environ.get("SPHERE_HELDOUT_FIT_EPOCHS", str(epochs)))
heldout_fit_lr = float(os.environ.get("SPHERE_HELDOUT_FIT_LR", str(lr)))

# Init models
model_geo = InverseGeodesicModel(
    num_trials=num_trials,
    latent_dim=latent_dim,
    n_neurons=N_neurons,
    solver=model_solver,
).to(device=device, dtype=TORCH_DTYPE)
model_free = InverseFreeModel(
    num_trials=num_trials,
    latent_dim=latent_dim,
    n_neurons=N_neurons,
    solver=model_solver,
).to(device=device, dtype=TORCH_DTYPE)

params_geo = count_parameters(model_geo)
params_free = count_parameters(model_free)
print(f"Geodesic Model Parameters: {params_geo}")
print(f"Free Dynamics Model Parameters: {params_free}\n")

# Train models
model_geo, loss_geo, nll_geo = train_and_evaluate(model_geo, train_dataset, t_eval, epochs=epochs, lr=lr)
print("-" * 20)
model_free, loss_free, nll_free = train_and_evaluate(model_free, train_dataset, t_eval, epochs=epochs, lr=lr)

rates_geo_pred, latents_geo = predict_all(model_geo, train_dataset, t_eval)
rates_free_pred, latents_free = predict_all(model_free, train_dataset, t_eval)
train_rates = rates_numpy(train_dataset)
heldout_rates = rates_numpy(heldout_dataset)
train_trial_indices = np.asarray([trial["idx"] for trial in train_dataset], dtype=np.int64)
heldout_trial_indices = np.asarray([trial["idx"] for trial in heldout_dataset], dtype=np.int64)
r_geo, r2_geo = corr_and_r2(train_rates, rates_geo_pred)
r_free, r2_free = corr_and_r2(train_rates, rates_free_pred)

if len(heldout_dataset) > 0:
    heldout_nll_geo, heldout_loss_geo = fit_heldout_initial_conditions(
        model_geo, heldout_dataset, t_eval, epochs=heldout_fit_epochs, lr=heldout_fit_lr
    )
    heldout_nll_free, heldout_loss_free = fit_heldout_initial_conditions(
        model_free, heldout_dataset, t_eval, epochs=heldout_fit_epochs, lr=heldout_fit_lr
    )
    rates_geo_heldout, latents_geo_heldout = predict_all(model_geo, heldout_dataset, t_eval)
    rates_free_heldout, latents_free_heldout = predict_all(model_free, heldout_dataset, t_eval)
    r_geo_heldout, r2_geo_heldout = corr_and_r2(heldout_rates, rates_geo_heldout)
    r_free_heldout, r2_free_heldout = corr_and_r2(heldout_rates, rates_free_heldout)
else:
    heldout_nll_geo = heldout_nll_free = np.nan
    heldout_loss_geo = heldout_loss_free = []
    rates_geo_heldout = rates_free_heldout = np.empty((0, n_timepoints, N_neurons), dtype=NP_DTYPE)
    latents_geo_heldout = latents_free_heldout = np.empty((0, n_timepoints, latent_dim), dtype=NP_DTYPE)
    r_geo_heldout = r2_geo_heldout = np.nan
    r_free_heldout = r2_free_heldout = np.nan

# Get AIC/BIC
aic_geo, bic_geo = calculate_ic(nll_geo, params_geo, num_obs)
aic_free, bic_free = calculate_ic(nll_free, params_free, num_obs)

print("\n" + "=" * 45)
print("            MODEL COMPARISON RESULTS ")
print("=" * 45)
print(f"{'Metric':<15} | {'Geodesic Model':<15} | {'Free Model':<15}")
print("-" * 48)
print(f"{'Parameters (k)':<15} | {params_geo:<15} | {params_free:<15}")
print(f"{'Train NLL':<15} | {nll_geo:<15.2f} | {nll_free:<15.2f}")
print(f"{'Heldout NLL':<15} | {heldout_nll_geo:<15.2f} | {heldout_nll_free:<15.2f}")
print(f"{'Train R2':<15} | {r2_geo:<15.4f} | {r2_free:<15.4f}")
print(f"{'Train r':<15} | {r_geo:<15.4f} | {r_free:<15.4f}")
print(f"{'Heldout R2':<15} | {r2_geo_heldout:<15.4f} | {r2_free_heldout:<15.4f}")
print(f"{'Heldout r':<15} | {r_geo_heldout:<15.4f} | {r_free_heldout:<15.4f}")
print(f"{'AIC':<15} | {aic_geo:<15.2f} | {aic_free:<15.2f}")
print(f"{'BIC':<15} | {bic_geo:<15.2f} | {bic_free:<15.2f}")
print("=" * 45)

best_aic = "Geodesic" if aic_geo < aic_free else "Free Dynamics"
best_bic = "Geodesic" if bic_geo < bic_free else "Free Dynamics"
print(f"\nPreferred Model by AIC: {best_aic}")
print(f"Preferred Model by BIC: {best_bic}")

plot_model_heatmap(train_rates, rates_geo_pred, rates_free_pred, dataset_idx=0, num_neurons=50)
plot_latents(true_latents, latents_geo, latents_free)
overlay_max_trials = int(os.environ.get("SPHERE_OVERLAY_MAX_TRIALS", "12"))
train_true_latents = true_latents[train_trial_indices]
plot_sphere_prediction_overlay(
    train_true_latents,
    rates_geo_pred,
    theta_centers,
    phi_centers,
    out_dir / "sphere_reconstruction_3d_overlay_geodesic.png",
    "True vs Predicted Sphere Trajectories (geodesic)",
    max_trials=overlay_max_trials,
)
plot_sphere_prediction_overlay(
    train_true_latents,
    rates_free_pred,
    theta_centers,
    phi_centers,
    out_dir / "sphere_reconstruction_3d_overlay_free.png",
    "True vs Predicted Sphere Trajectories (free)",
    max_trials=overlay_max_trials,
)
plot_sphere_prediction_comparison(
    train_true_latents,
    rates_geo_pred,
    rates_free_pred,
    theta_centers,
    phi_centers,
    out_dir / "sphere_reconstruction_3d_overlay_all.png",
    "train",
    max_trials=overlay_max_trials,
)
if len(heldout_dataset) > 0:
    heldout_true_latents = true_latents[heldout_trial_indices]
    plot_sphere_prediction_overlay(
        heldout_true_latents,
        rates_geo_heldout,
        theta_centers,
        phi_centers,
        out_dir / "sphere_heldout_reconstruction_3d_overlay_geodesic.png",
        "True vs Predicted Sphere Trajectories (heldout geodesic)",
        max_trials=overlay_max_trials,
    )
    plot_sphere_prediction_overlay(
        heldout_true_latents,
        rates_free_heldout,
        theta_centers,
        phi_centers,
        out_dir / "sphere_heldout_reconstruction_3d_overlay_free.png",
        "True vs Predicted Sphere Trajectories (heldout free)",
        max_trials=overlay_max_trials,
    )
    plot_sphere_prediction_comparison(
        heldout_true_latents,
        rates_geo_heldout,
        rates_free_heldout,
        theta_centers,
        phi_centers,
        out_dir / "sphere_heldout_reconstruction_3d_overlay_all.png",
        "heldout",
        max_trials=overlay_max_trials,
    )

summary = {
    "config": {
        "num_trials": num_trials,
        "num_train_trials": len(train_dataset),
        "num_heldout_trials": len(heldout_dataset),
        "heldout_frac": heldout_frac,
        "heldout_fit_epochs": heldout_fit_epochs,
        "heldout_fit_lr": heldout_fit_lr,
        "N_neurons": N_neurons,
        "n_timepoints": n_timepoints,
        "latent_dim": latent_dim,
        "kappa": kappa,
        "speed": speed,
        "epochs": epochs,
        "lr": lr,
        "device": str(device),
        "dtype": str(TORCH_DTYPE),
        "model_solver": model_solver,
    },
    "geodesic": {
        "params": params_geo,
        "final_nll": nll_geo,
        "train_nll": nll_geo,
        "heldout_nll": heldout_nll_geo,
        "R2": r2_geo,
        "r": r_geo,
        "train_R2": r2_geo,
        "train_r": r_geo,
        "heldout_R2": r2_geo_heldout,
        "heldout_r": r_geo_heldout,
        "AIC": aic_geo,
        "BIC": bic_geo,
    },
    "free": {
        "params": params_free,
        "final_nll": nll_free,
        "train_nll": nll_free,
        "heldout_nll": heldout_nll_free,
        "R2": r2_free,
        "r": r_free,
        "train_R2": r2_free,
        "train_r": r_free,
        "heldout_R2": r2_free_heldout,
        "heldout_r": r_free_heldout,
        "AIC": aic_free,
        "BIC": bic_free,
    },
    "preferred_by_AIC": best_aic,
    "preferred_by_BIC": best_bic,
}
# writing to a directory here so I can look back through the previous runs 
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
torch.save(
    {
        "config": summary["config"],
        "geodesic_state_dict": model_geo.state_dict(),
        "free_state_dict": model_free.state_dict(),
        "train_trial_indices": train_trial_indices,
        "heldout_trial_indices": heldout_trial_indices,
    },
    out_dir / "model_checkpoint.pt",
)
np.savez_compressed(
    out_dir / "fit_outputs.npz",
    activities=activities.astype(NP_DTYPE),
    train_rates=train_rates.astype(NP_DTYPE),
    heldout_rates=heldout_rates.astype(NP_DTYPE),
    train_trial_indices=train_trial_indices,
    heldout_trial_indices=heldout_trial_indices,
    rates_geo_pred=rates_geo_pred.astype(NP_DTYPE),
    rates_free_pred=rates_free_pred.astype(NP_DTYPE),
    rates_geo_heldout=rates_geo_heldout.astype(NP_DTYPE),
    rates_free_heldout=rates_free_heldout.astype(NP_DTYPE),
    true_latents=true_latents.astype(NP_DTYPE),
    latents_geo=latents_geo.astype(NP_DTYPE),
    latents_free=latents_free.astype(NP_DTYPE),
    latents_geo_heldout=latents_geo_heldout.astype(NP_DTYPE),
    latents_free_heldout=latents_free_heldout.astype(NP_DTYPE),
    theta_centers=theta_centers.astype(NP_DTYPE),
    phi_centers=phi_centers.astype(NP_DTYPE),
    loss_geo=np.asarray(loss_geo, dtype=NP_DTYPE),
    loss_free=np.asarray(loss_free, dtype=NP_DTYPE),
    heldout_loss_geo=np.asarray(heldout_loss_geo, dtype=NP_DTYPE),
    heldout_loss_free=np.asarray(heldout_loss_free, dtype=NP_DTYPE),
)
print(f"Saved outputs to {out_dir}")
