#!/usr/bin/env python3
"""Fit the multi-trial geodesic and free-dynamics models to E65.

This is the E65 counterpart of ``fit_sphere_trials.py``.  It keeps the same
learned-metric geodesic model, free second-order dynamics comparison, shared
neural decoder, and per-trial initial latent position/velocity.  The synthetic
sphere generator is replaced by raw E65 trials.

E65 trials have different numbers of frames.  They are padded only to batch
them efficiently; padded entries are excluded from every loss and metric.
Neither position nor any other behavioral variable is given to the models.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_dtype():
    name = os.environ.get("E65_DTYPE", "float32").strip().lower()
    if name == "float32":
        return torch.float32, np.float32
    if name == "float64":
        return torch.float64, np.float64
    raise ValueError("E65_DTYPE must be 'float32' or 'float64'.")


TORCH_DTYPE, NP_DTYPE = configure_dtype()
torch.set_default_dtype(TORCH_DTYPE)


def choose_device() -> torch.device:
    requested = os.environ.get("E65_DEVICE", os.environ.get("GEODESIC_DEVICE"))
    if requested:
        selected = torch.device(requested)
    elif torch.cuda.is_available():
        selected = torch.device("cuda")
    else:
        selected = torch.device("cpu")
    if selected.type == "mps" and TORCH_DTYPE == torch.float64:
        raise RuntimeError("PyTorch MPS does not support float64, so please use E65_DTYPE=float32 or E65_DEVICE=cpu.")
    return selected


class MetricNetwork(nn.Module):
    """Map a latent coordinate to a positive-definite learned metric tensor."""

    def __init__(self, latent_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        out_dim = latent_dim * (latent_dim + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.eps = 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.net(x)
        lower = torch.zeros(
            batch_size,
            self.latent_dim,
            self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                value = torch.exp(out[:, idx]) if i == j else out[:, idx]
                lower[:, i, j] = value
                idx += 1
        identity = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        return torch.bmm(lower, lower.transpose(1, 2)) + self.eps * identity


class GeodesicDynamics(nn.Module):
    """Second-order geodesic dynamics induced by the learned metric."""

    def __init__(self, metric_net: MetricNetwork):
        super().__init__()
        self.metric_net = metric_net

    def compute_christoffel(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        metric = self.metric_net(x)
        batch_size, latent_dim, _ = metric.shape

        metric_derivative = torch.zeros(
            batch_size,
            latent_dim,
            latent_dim,
            latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        # The metric is symmetric, so its derivatives are symmetric in i/j as
        # well.  Computing only the lower triangle reduces d^2 autograd calls
        # to d(d+1)/2 without changing the metric or geodesic equation.
        for i in range(latent_dim):
            for j in range(i + 1):
                derivative = torch.autograd.grad(
                    outputs=metric[:, i, j].sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                metric_derivative[:, i, j, :] = derivative
                if i != j:
                    metric_derivative[:, j, i, :] = derivative

        inverse_metric = torch.linalg.inv(metric)
        # Gamma^k_mn = 1/2 g^kl (d_m g_nl + d_n g_ml - d_l g_mn).
        # These permutations form the first-kind Christoffel tensor with axes
        # [batch, ell, m, n]; einsum then raises its first index.  This is the
        # same contraction as the original four nested Python loops.
        first_kind = 0.5 * (
            metric_derivative.permute(0, 2, 3, 1)
            + metric_derivative.permute(0, 2, 1, 3)
            - metric_derivative.permute(0, 3, 1, 2)
        )
        return torch.einsum("bkl,blmn->bkmn", inverse_metric, first_kind)

    def acceleration(self, x: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        christoffel = self.compute_christoffel(x)
        return -torch.einsum("bkmn,bm,bn->bk", christoffel, velocity, velocity)

    def euler_step(self, x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor):
        acceleration = self.acceleration(x, velocity)
        return x + velocity * dt, velocity + acceleration * dt

    def rk4_step(self, x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor):
        k1_v = self.acceleration(x, velocity)
        k1_x = velocity
        k2_v = self.acceleration(x + 0.5 * dt * k1_x, velocity + 0.5 * dt * k1_v)
        k2_x = velocity + 0.5 * dt * k1_v
        k3_v = self.acceleration(x + 0.5 * dt * k2_x, velocity + 0.5 * dt * k2_v)
        k3_x = velocity + 0.5 * dt * k2_v
        k4_v = self.acceleration(x + dt * k3_x, velocity + dt * k3_v)
        k4_x = velocity + dt * k3_v
        next_velocity = velocity + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        next_x = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        return next_x, next_velocity

    def forward(self, state, dt: torch.Tensor, solver: str = "rk4"):
        x, velocity = state
        if solver == "euler":
            return self.euler_step(x, velocity, dt)
        return self.rk4_step(x, velocity, dt)


class FreeDynamics(nn.Module):
    """Unconstrained second-order dynamics used as the comparison model."""

    def __init__(self, latent_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def acceleration(self, x: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, velocity], dim=-1))

    def euler_step(self, x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor):
        acceleration = self.acceleration(x, velocity)
        return x + velocity * dt, velocity + acceleration * dt

    def rk4_step(self, x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor):
        k1_v = self.acceleration(x, velocity)
        k1_x = velocity
        k2_v = self.acceleration(x + 0.5 * dt * k1_x, velocity + 0.5 * dt * k1_v)
        k2_x = velocity + 0.5 * dt * k1_v
        k3_v = self.acceleration(x + 0.5 * dt * k2_x, velocity + 0.5 * dt * k2_v)
        k3_x = velocity + 0.5 * dt * k2_v
        k4_v = self.acceleration(x + dt * k3_x, velocity + dt * k3_v)
        k4_x = velocity + dt * k3_v
        next_velocity = velocity + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        next_x = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        return next_x, next_velocity

    def forward(self, state, dt: torch.Tensor, solver: str = "rk4"):
        x, velocity = state
        if solver == "euler":
            return self.euler_step(x, velocity, dt)
        return self.rk4_step(x, velocity, dt)


class NeuralDecoder(nn.Module):
    def __init__(self, latent_dim: int, n_neurons: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_neurons),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryModel(nn.Module):
    """Shared rollout/decoder code for geodesic and free dynamics."""

    def __init__(
        self,
        num_trials: int,
        latent_dim: int,
        n_neurons: int,
        dynamics: nn.Module,
        solver: str,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        self.solver = solver
        self.x0 = nn.Parameter(torch.randn(num_trials, latent_dim))
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim))

    def forward(self, trial_indices: torch.Tensor, time_steps: torch.Tensor):
        """Roll out trials using their real E65 frame-to-frame intervals.

        ``time_steps`` has shape [batch, sequence_length - 1].  Padded time
        steps are zero, so padded trajectories remain frozen and are masked
        out of the objective by the caller.
        """
        trial_indices = trial_indices.to(device=self.x0.device, dtype=torch.long)
        time_steps = time_steps.to(device=self.x0.device, dtype=self.x0.dtype)
        x = self.x0[trial_indices]
        velocity = self.v0[trial_indices]
        latents = [x]
        for step in range(time_steps.shape[1]):
            dt = time_steps[:, step : step + 1]
            x, velocity = self.dynamics((x, velocity), dt, solver=self.solver)
            latents.append(x)
        latent_path = torch.stack(latents, dim=1)
        flat_rates = self.decoder(latent_path.reshape(-1, latent_path.shape[-1]))
        rates = flat_rates.reshape(latent_path.shape[0], latent_path.shape[1], -1)
        return latent_path, rates


class InverseGeodesicModel(TrajectoryModel):
    def __init__(self, num_trials: int, latent_dim: int, n_neurons: int, solver: str):
        metric = MetricNetwork(latent_dim)
        super().__init__(
            num_trials,
            latent_dim,
            n_neurons,
            GeodesicDynamics(metric),
            solver,
        )
        self.metric = metric


class InverseFreeModel(TrajectoryModel):
    def __init__(self, num_trials: int, latent_dim: int, n_neurons: int, solver: str):
        super().__init__(
            num_trials,
            latent_dim,
            n_neurons,
            FreeDynamics(latent_dim),
            solver,
        )


def load_e65(npz_path: Path):
    with np.load(npz_path) as source:
        raw_rates = np.asarray(source["roi"], dtype=NP_DTYPE).T
        trial_labels = np.asarray(source["Trial"]).reshape(-1)
        frame_times = np.asarray(source["Time"], dtype=NP_DTYPE).reshape(-1)
    if not (len(raw_rates) == len(trial_labels) == len(frame_times)):
        raise ValueError("roi, Trial, and Time must have the same frame count.")
    if not np.isfinite(raw_rates).all() or not np.isfinite(frame_times).all():
        raise ValueError("E65 arrays contain NaN or infinite values.")
    if np.any(raw_rates < 0):
        raise ValueError("Poisson-rate decoding requires nonnegative E65 activity.")
    return raw_rates, trial_labels, frame_times


def split_trials(trial_ids: np.ndarray, heldout_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(trial_ids)) # randomly reorder trial indices (splits whole trials, not frames)
    if heldout_fraction <= 0.0 or len(trial_ids) == 1:
        return shuffled, np.empty(0, dtype=np.int64)
    n_heldout = max(1, int(round(heldout_fraction * len(trial_ids))))
    n_heldout = min(n_heldout, len(trial_ids) - 1)
    return shuffled[n_heldout:], shuffled[:n_heldout]


def prepare_e65_data(
    raw_rates: np.ndarray,
    trial_labels: np.ndarray,
    frame_times: np.ndarray,
    heldout_fraction: float,
    seed: int,
    max_trials: int,
    active_threshold: float,
    filter_train_silent: bool,
):
    trial_ids = np.unique(trial_labels)
    rng = np.random.default_rng(seed)
    if max_trials > 0 and max_trials < len(trial_ids):
        trial_ids = np.sort(rng.choice(trial_ids, size=max_trials, replace=False))

    train_indices, heldout_indices = split_trials(trial_ids, heldout_fraction, seed)
    train_ids = trial_ids[train_indices]
    train_frame_mask = np.isin(trial_labels, train_ids)
    globally_active = np.any(raw_rates > active_threshold, axis=0)
    train_active = np.any(raw_rates[train_frame_mask] > active_threshold, axis=0)
    neuron_mask = train_active if filter_train_silent else globally_active
    if not np.any(neuron_mask):
        raise ValueError("The neuron filter removed every neuron.")

    trial_rates = []
    trial_times = []
    for trial_id in trial_ids:
        frame_mask = trial_labels == trial_id
        rates = raw_rates[frame_mask][:, neuron_mask]
        times = frame_times[frame_mask]
        order = np.argsort(times, kind="stable")
        rates = rates[order]
        times = times[order] - times[order][0]
        differences = np.diff(times)
        if np.any(differences <= 0):
            positive = differences[differences > 0]
            replacement = float(np.median(positive)) if len(positive) else 1.0
            differences = np.where(differences > 0, differences, replacement)
            times = np.concatenate([[0.0], np.cumsum(differences)]).astype(NP_DTYPE)
        trial_rates.append(rates.astype(NP_DTYPE, copy=False))
        trial_times.append(times.astype(NP_DTYPE, copy=False))

    max_length = max(len(values) for values in trial_rates)
    n_neurons = int(neuron_mask.sum())
    padded_rates = np.zeros((len(trial_ids), max_length, n_neurons), dtype=NP_DTYPE)
    padded_times = np.zeros((len(trial_ids), max_length), dtype=NP_DTYPE)
    valid_mask = np.zeros((len(trial_ids), max_length), dtype=bool)
    time_steps = np.zeros((len(trial_ids), max_length - 1), dtype=NP_DTYPE)
    lengths = np.zeros(len(trial_ids), dtype=np.int64)
    for idx, (rates, times) in enumerate(zip(trial_rates, trial_times)):
        length = len(rates)
        lengths[idx] = length
        padded_rates[idx, :length] = rates
        padded_times[idx, :length] = times
        valid_mask[idx, :length] = True
        time_steps[idx, : length - 1] = np.diff(times)

    metadata = {
        "trial_ids": trial_ids,
        "train_indices": train_indices.astype(np.int64),
        "heldout_indices": heldout_indices.astype(np.int64),
        "neuron_mask": neuron_mask,
        "globally_active": globally_active,
        "train_active": train_active,
        "lengths": lengths,
    }
    arrays = {
        "rates": torch.tensor(padded_rates, dtype=TORCH_DTYPE),
        "times": padded_times,
        "mask": torch.tensor(valid_mask, dtype=torch.bool),
        "time_steps": torch.tensor(time_steps, dtype=TORCH_DTYPE),
    }
    return arrays, metadata


def iter_index_batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng):
    ordered = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def batch_tensors(arrays, batch_indices: np.ndarray, selected_device: torch.device, max_steps=None):
    index_tensor = torch.tensor(batch_indices, dtype=torch.long)
    lengths = arrays["mask"][index_tensor].sum(dim=1)
    sequence_length = int(lengths.max().item())
    if max_steps is not None:
        sequence_length = min(sequence_length, int(max_steps))
    rates = arrays["rates"][index_tensor, :sequence_length].to(selected_device)
    mask = arrays["mask"][index_tensor, :sequence_length].to(selected_device)
    time_steps = arrays["time_steps"][index_tensor, : max(0, sequence_length - 1)].to(selected_device)
    trial_indices = index_tensor.to(selected_device)
    return trial_indices, rates, mask, time_steps


def masked_nll(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pointwise = prediction - target * torch.log(prediction + 1e-8)
    return (pointwise * mask.unsqueeze(-1)).sum()


def masked_statistics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    valid = mask.unsqueeze(-1).expand_as(target)
    errors = (prediction - target)[valid]
    values = target[valid]
    return (
        float(torch.sum(errors * errors).detach().cpu()),
        float(values.sum().detach().cpu()),
        float(torch.sum(values * values).detach().cpu()),
        int(values.numel()),
    )

#=======Asked Codex to add in checkpointing functionality========#
def atomic_torch_save(payload, checkpoint_path: Path):
    """Write a checkpoint atomically so interruption cannot corrupt the last save."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_torch_checkpoint(checkpoint_path: Path, selected_device: torch.device):
    """Load checkpoints on both current and older PyTorch releases."""
    try:
        return torch.load(
            checkpoint_path,
            map_location=selected_device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=selected_device)


def restore_training_checkpoint(
    checkpoint_path: Path,
    stage_name: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    selected_device: torch.device,
    resume: bool,
):
    if not resume or not checkpoint_path.exists():
        return 0, None, None
    checkpoint = load_torch_checkpoint(checkpoint_path, selected_device)
    saved_stage = checkpoint.get("stage")
    if saved_stage != stage_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} belongs to stage {saved_stage!r}, "
            f"not {stage_name!r}."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["next_epoch"])
    print(f"Resuming {stage_name} at epoch {start_epoch} from {checkpoint_path}")
    return start_epoch, checkpoint.get("history"), checkpoint.get("numpy_rng_state")


def save_training_checkpoint(
    checkpoint_path: Path,
    stage_name: str,
    next_epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    history,
    numpy_rng_state=None,
):
    atomic_torch_save(
        {
            "stage": stage_name,
            "next_epoch": int(next_epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "numpy_rng_state": numpy_rng_state,
        },
        checkpoint_path,
    )
#===========End checkpointing functionality==================#

def train_model(
    model: nn.Module,
    arrays,
    train_indices: np.ndarray,
    selected_device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    log_every: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
    stage_name: str,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = {"epoch": [], "nll": [], "mse": [], "r2": []}
    start_epoch, saved_history, saved_rng_state = restore_training_checkpoint(
        checkpoint_path,
        stage_name,
        model,
        optimizer,
        selected_device,
        resume,
    )
    if saved_history is not None:
        history = saved_history
    if saved_rng_state is not None:
        rng.bit_generator.state = saved_rng_state
    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is already at epoch {start_epoch}, "
            f"which exceeds requested E65_EPOCHS={epochs}."
        )

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        nll_total = 0.0
        squared_error_total = 0.0
        target_sum = 0.0
        target_square_sum = 0.0
        observation_count = 0
        for batch_indices in iter_index_batches(train_indices, batch_size, True, rng):
            trial_indices, target, mask, time_steps = batch_tensors(
                arrays, batch_indices, selected_device
            )
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model(trial_indices, time_steps)
            nll = masked_nll(prediction, target, mask)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            stats = masked_statistics(prediction.detach(), target, mask)
            optimizer.step()

            nll_total += float(nll.detach().cpu())
            squared_error_total += stats[0]
            target_sum += stats[1]
            target_square_sum += stats[2]
            observation_count += stats[3]

        target_ss = target_square_sum - target_sum * target_sum / max(observation_count, 1)
        mse = squared_error_total / max(observation_count, 1)
        r2 = 1.0 - squared_error_total / target_ss if target_ss > 0 else float("nan")
        history["epoch"].append(epoch)
        history["nll"].append(nll_total)
        history["mse"].append(mse)
        history["r2"].append(r2)
        if checkpoint_every > 0 and (
            (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1
        ):
            save_training_checkpoint(
                checkpoint_path,
                stage_name,
                epoch + 1,
                model,
                optimizer,
                history,
                rng.bit_generator.state,
            )
        if epoch % log_every == 0 or epoch == epochs - 1:
            epoch_seconds = time.perf_counter() - epoch_started
            print(
                f"Epoch {epoch:03d} | NLL Sum: {nll_total:.2f} | "
                f"MSE: {mse:.8f} | R2: {r2:.6f} | Seconds: {epoch_seconds:.1f}"
            )
    return history


def fit_context_initial_conditions(
    model: nn.Module,
    arrays,
    heldout_indices: np.ndarray,
    selected_device: torch.device,
    context_steps: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    log_every: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
    stage_name: str,
):
    if len(heldout_indices) == 0:
        return []
    prior_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in {"x0", "v0"})
    optimizer = optim.Adam([model.x0, model.v0], lr=learning_rate)
    history = []
    start_epoch, saved_history, _ = restore_training_checkpoint(
        checkpoint_path,
        stage_name,
        model,
        optimizer,
        selected_device,
        resume,
    )
    if saved_history is not None:
        history = saved_history
    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is already at epoch {start_epoch}, "
            f"which exceeds requested E65_CONTEXT_EPOCHS={epochs}."
        )
    try:
        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            epoch_nll = 0.0
            for batch_indices in iter_index_batches(
                heldout_indices, batch_size, False, np.random.default_rng(0)
            ):
                trial_indices, target, mask, time_steps = batch_tensors(
                    arrays, batch_indices, selected_device, max_steps=context_steps
                )
                optimizer.zero_grad(set_to_none=True)
                _, prediction = model(trial_indices, time_steps)
                nll = masked_nll(prediction, target, mask)
                nll.backward()
                torch.nn.utils.clip_grad_norm_([model.x0, model.v0], max_norm=1.0)
                optimizer.step()
                epoch_nll += float(nll.detach().cpu())
            history.append(epoch_nll)
            if checkpoint_every > 0 and (
                (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1
            ):
                save_training_checkpoint(
                    checkpoint_path,
                    stage_name,
                    epoch + 1,
                    model,
                    optimizer,
                    history,
                )
            if epoch % log_every == 0 or epoch == epochs - 1:
                # logging the time taken per epoch to see correlation with NLL sum
                epoch_seconds = time.perf_counter() - epoch_started
                print(
                    f"Context epoch {epoch:03d} | NLL Sum: {epoch_nll:.2f} | "
                    f"Seconds: {epoch_seconds:.1f}"
                )
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(prior_requires_grad[name])
    return history


def predict_padded(
    model: nn.Module,
    arrays,
    indices: np.ndarray,
    selected_device: torch.device,
    batch_size: int,
    n_trials: int,
    max_length: int,
    n_neurons: int,
    latent_dim: int,
):
    """
    Runs the trained model on multiple variable-length trials 
    E65 trials have different lengths, so using NaN entries are placeholders
    (excluded from evaluation)
    """
    
    rate_output = np.full((n_trials, max_length, n_neurons), np.nan, dtype=NP_DTYPE)
    latent_output = np.full((n_trials, max_length, latent_dim), np.nan, dtype=NP_DTYPE)
    for batch_indices in iter_index_batches(indices, batch_size, False, np.random.default_rng(0)):
        trial_indices, _, mask, time_steps = batch_tensors(arrays, batch_indices, selected_device)
        with torch.enable_grad():
            latents, predictions = model(trial_indices, time_steps)
        lengths = mask.sum(dim=1).detach().cpu().numpy().astype(int)
        predictions = predictions.detach().cpu().numpy()
        latents = latents.detach().cpu().numpy()
        for row, trial_index in enumerate(batch_indices):
            length = lengths[row]
            rate_output[trial_index, :length] = predictions[row, :length]
            latent_output[trial_index, :length] = latents[row, :length]
    return rate_output, latent_output


def score_predictions(true_rates, predicted_rates, valid_mask, indices, start_step=0):
    true_values = []
    predicted_values = []
    for trial_index in indices:
        valid_steps = np.flatnonzero(valid_mask[trial_index])
        valid_steps = valid_steps[valid_steps >= start_step]
        if len(valid_steps) == 0:
            continue
        true_values.append(true_rates[trial_index, valid_steps].reshape(-1))
        predicted_values.append(predicted_rates[trial_index, valid_steps].reshape(-1))
    if not true_values:
        return {"nll": float("nan"), "mse": float("nan"), "r2": float("nan"), "r": float("nan"), "n": 0}
    target = np.concatenate(true_values).astype(np.float64)
    prediction = np.concatenate(predicted_values).astype(np.float64)
    residual = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    target_std = target.std()
    prediction_std = prediction.std()
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target_std > 0 and prediction_std > 0
        else float("nan")
    )
    return {
        "nll": float(np.sum(prediction - target * np.log(prediction + 1e-8))),
        "mse": float(np.mean(residual * residual)),
        "r2": float(1.0 - np.sum(residual * residual) / denominator) if denominator > 0 else float("nan"),
        "r": correlation,
        "n": int(target.size),
    }


def count_training_parameters(model: nn.Module, n_trials: int, n_train: int, latent_dim: int):
    total = sum(parameter.numel() for parameter in model.parameters())
    all_initial_conditions = 2 * n_trials * latent_dim
    trained_initial_conditions = 2 * n_train * latent_dim
    return total - all_initial_conditions + trained_initial_conditions


def information_criteria(nll: float, n_parameters: int, n_observations: int):
    aic = 2 * n_parameters + 2 * nll
    bic = n_parameters * np.log(n_observations) + 2 * nll
    return float(aic), float(bic)


def plot_reconstruction(
    true_rates,
    geodesic_rates,
    free_rates,
    times,
    length,
    trial_id,
    output_path: Path,
    n_neurons=50,
):
    n_neurons = min(n_neurons, true_rates.shape[-1])
    true_matrix = true_rates[:length, :n_neurons].T
    geodesic_matrix = geodesic_rates[:length, :n_neurons].T
    free_matrix = free_rates[:length, :n_neurons].T
    values = np.concatenate([true_matrix.ravel(), geodesic_matrix.ravel(), free_matrix.ravel()])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    extent = [float(times[0]), float(times[length - 1]), 0, n_neurons]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    titles = ["E65 activity", "Geodesic prediction", "Free prediction"]
    for axis, matrix, title in zip(axes, [true_matrix, geodesic_matrix, free_matrix], titles):
        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
    axes[0].set_ylabel("Neuron")
    fig.suptitle(f"Trial {trial_id:g}")
    fig.colorbar(image, ax=axes, label="Activity", shrink=0.8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_histories(geodesic_history, free_history, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric, title in zip(axes, ["nll", "mse", "r2"], ["NLL", "MSE", "R²"]):
        axis.plot(geodesic_history["epoch"], geodesic_history[metric], label="Geodesic")
        axis.plot(free_history["epoch"], free_history[metric], label="Free")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    seed = int(os.environ.get("E65_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Hyperparameter selection from input
    selected_device = choose_device()
    data_path = Path(os.environ.get("E65_NPZ_PATH", "src/npz_e65_data/E65_data.npz")).expanduser()
    output_dir = Path(os.environ.get("E65_OUT_DIR", "runs/geodesic_e65_trials")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dim = int(os.environ.get("E65_LATENT_DIM", "5"))
    epochs = int(os.environ.get("E65_EPOCHS", "300"))
    learning_rate = float(os.environ.get("E65_LR", "0.001"))
    heldout_fraction = float(os.environ.get("E65_HELDOUT_FRAC", "0.2"))
    context_steps = int(os.environ.get("E65_CONTEXT_STEPS", "10"))
    context_epochs = int(os.environ.get("E65_CONTEXT_EPOCHS", str(epochs)))
    context_learning_rate = float(os.environ.get("E65_CONTEXT_LR", str(learning_rate)))
    batch_size = int(os.environ.get("E65_BATCH_SIZE", "16"))
    max_trials = int(os.environ.get("E65_MAX_TRIALS", "0"))
    active_threshold = float(os.environ.get("E65_ACTIVE_THRESHOLD", "0.0"))
    filter_train_silent = env_flag("E65_FILTER_TRAIN_SILENT", True)
    solver = os.environ.get("E65_MODEL_SOLVER", "rk4").strip().lower()
    log_every = int(os.environ.get("E65_LOG_EVERY", "10"))
    checkpoint_every = int(os.environ.get("E65_CHECKPOINT_EVERY", "1"))
    resume_checkpoints = env_flag("E65_RESUME", True)
    checkpoint_dir = Path(
        os.environ.get("E65_CHECKPOINT_DIR", str(output_dir / "checkpoints"))
    ).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Error checking for hyperparamters (only the major ones)
    if solver not in {"rk4", "euler"}:
        raise ValueError("E65_MODEL_SOLVER must be 'rk4' or 'euler'.")
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("E65_HELDOUT_FRAC must be in [0, 1).")
    if checkpoint_every < 0:
        raise ValueError("E65_CHECKPOINT_EVERY must be nonnegative.")

    print(f"Using device: {selected_device}")
    print(f"Using dtype: {TORCH_DTYPE}")
    print(
        f"Checkpoints: {checkpoint_dir} "
        f"(every {checkpoint_every} epoch(s), resume={resume_checkpoints})"
    )
    print(f"Loading E65: {data_path}")
    raw_rates, trial_labels, frame_times = load_e65(data_path)
    arrays, metadata = prepare_e65_data(
        raw_rates,
        trial_labels,
        frame_times,
        heldout_fraction,
        seed,
        max_trials,
        active_threshold,
        filter_train_silent,
    )

    trial_ids = metadata["trial_ids"]
    train_indices = metadata["train_indices"]
    heldout_indices = metadata["heldout_indices"]
    neuron_mask = metadata["neuron_mask"]
    lengths = metadata["lengths"]
    num_trials = len(trial_ids)
    n_neurons = int(neuron_mask.sum())
    max_length = arrays["rates"].shape[1]
    if len(heldout_indices) and not 2 <= context_steps < int(lengths[heldout_indices].min()):
        raise ValueError(
            "E65_CONTEXT_STEPS must be at least 2 and smaller than every heldout trial. "
            f"Shortest heldout trial has {int(lengths[heldout_indices].min())} frames."
        )

    print(
        f"Prepared {num_trials} trials: train={len(train_indices)}, heldout={len(heldout_indices)}, "
        f"frames={int(lengths.sum())}, length={lengths.min()}-{lengths.max()}, neurons={n_neurons}."
    )
    print(
        f"Neuron filter removed {raw_rates.shape[1] - n_neurons}; "
        f"globally silent={raw_rates.shape[1] - int(metadata['globally_active'].sum())}, "
        f"train-silent={raw_rates.shape[1] - int(metadata['train_active'].sum())}."
    )
    print("Inputs: raw neural activity and raw within-trial frame times only; no behavior or position.")
    if len(heldout_indices):
        print(
            f"Heldout protocol: fit x0/v0 on {context_steps} prefix frames, then forecast each remaining suffix."
        )

    model_geodesic = InverseGeodesicModel(
        num_trials, latent_dim, n_neurons, solver
    ).to(device=selected_device, dtype=TORCH_DTYPE)
    model_free = InverseFreeModel(
        num_trials, latent_dim, n_neurons, solver
    ).to(device=selected_device, dtype=TORCH_DTYPE)
    params_geodesic = count_training_parameters(
        model_geodesic, num_trials, len(train_indices), latent_dim
    )
    params_free = count_training_parameters(model_free, num_trials, len(train_indices), latent_dim)
    print(f"Geodesic training parameters: {params_geodesic}")
    print(f"Free training parameters: {params_free}\n")

    print("Training geodesic model...")
    geodesic_history = train_model(
        model_geodesic,
        arrays,
        train_indices,
        selected_device,
        epochs,
        learning_rate,
        batch_size,
        seed,
        log_every,
        checkpoint_dir / "geodesic_training.pt",
        checkpoint_every,
        resume_checkpoints,
        "geodesic_training",
    )
    print("Training free-dynamics model...")
    free_history = train_model(
        model_free,
        arrays,
        train_indices,
        selected_device,
        epochs,
        learning_rate,
        batch_size,
        seed,
        log_every,
        checkpoint_dir / "free_training.pt",
        checkpoint_every,
        resume_checkpoints,
        "free_training",
    )

    if len(heldout_indices):
        print("Fitting heldout geodesic x0/v0 from context only...")
        geodesic_context_history = fit_context_initial_conditions(
            model_geodesic,
            arrays,
            heldout_indices,
            selected_device,
            context_steps,
            context_epochs,
            context_learning_rate,
            batch_size,
            log_every,
            checkpoint_dir / "geodesic_context.pt",
            checkpoint_every,
            resume_checkpoints,
            "geodesic_context",
        )
        print("Fitting heldout free x0/v0 from context only...")
        free_context_history = fit_context_initial_conditions(
            model_free,
            arrays,
            heldout_indices,
            selected_device,
            context_steps,
            context_epochs,
            context_learning_rate,
            batch_size,
            log_every,
            checkpoint_dir / "free_context.pt",
            checkpoint_every,
            resume_checkpoints,
            "free_context",
        )
    else:
        geodesic_context_history = []
        free_context_history = []

    all_indices = np.arange(num_trials, dtype=np.int64)
    geodesic_predictions, geodesic_latents = predict_padded(
        model_geodesic,
        arrays,
        all_indices,
        selected_device,
        batch_size,
        num_trials,
        max_length,
        n_neurons,
        latent_dim,
    )
    free_predictions, free_latents = predict_padded(
        model_free,
        arrays,
        all_indices,
        selected_device,
        batch_size,
        num_trials,
        max_length,
        n_neurons,
        latent_dim,
    )
    true_rates = arrays["rates"].numpy()
    valid_mask = arrays["mask"].numpy()
    train_geodesic = score_predictions(
        true_rates, geodesic_predictions, valid_mask, train_indices
    )
    train_free = score_predictions(true_rates, free_predictions, valid_mask, train_indices)
    forecast_geodesic = score_predictions(
        true_rates, geodesic_predictions, valid_mask, heldout_indices, context_steps
    )
    forecast_free = score_predictions(
        true_rates, free_predictions, valid_mask, heldout_indices, context_steps
    )

    aic_geodesic, bic_geodesic = information_criteria(
        train_geodesic["nll"], params_geodesic, train_geodesic["n"]
    )
    aic_free, bic_free = information_criteria(
        train_free["nll"], params_free, train_free["n"]
    )

    print("\n" + "=" * 58)
    print(" E65 MODEL COMPARISON")
    print("=" * 58)
    print(f"{'Metric':<18} | {'Geodesic':>16} | {'Free':>16}")
    print("-" * 58)
    print(f"{'Parameters':<18} | {params_geodesic:>16d} | {params_free:>16d}")
    print(f"{'Train NLL':<18} | {train_geodesic['nll']:>16.2f} | {train_free['nll']:>16.2f}")
    print(f"{'Train MSE':<18} | {train_geodesic['mse']:>16.8f} | {train_free['mse']:>16.8f}")
    print(f"{'Train R2':<18} | {train_geodesic['r2']:>16.4f} | {train_free['r2']:>16.4f}")
    print(f"{'Train r':<18} | {train_geodesic['r']:>16.4f} | {train_free['r']:>16.4f}")
    print(f"{'Forecast R2':<18} | {forecast_geodesic['r2']:>16.4f} | {forecast_free['r2']:>16.4f}")
    print(f"{'Forecast r':<18} | {forecast_geodesic['r']:>16.4f} | {forecast_free['r']:>16.4f}")
    print(f"{'AIC':<18} | {aic_geodesic:>16.2f} | {aic_free:>16.2f}")
    print(f"{'BIC':<18} | {bic_geodesic:>16.2f} | {bic_free:>16.2f}")
    print("=" * 58)

    plot_index = int(heldout_indices[0] if len(heldout_indices) else train_indices[0])
    plot_reconstruction(
        true_rates[plot_index],
        geodesic_predictions[plot_index],
        free_predictions[plot_index],
        arrays["times"][plot_index],
        int(lengths[plot_index]),
        trial_ids[plot_index],
        output_dir / "e65_true_vs_geodesic_vs_free.png",
    )
    plot_histories(geodesic_history, free_history, output_dir / "training_metrics.png")

    summary = {
        "config": {
            "data_path": str(data_path),
            "num_trials": num_trials,
            "num_train_trials": len(train_indices),
            "num_heldout_trials": len(heldout_indices),
            "heldout_fraction": heldout_fraction,
            "heldout_evaluation": "context_prefix_fit_then_suffix_forecast",
            "context_steps": context_steps,
            "context_epochs": context_epochs,
            "n_neurons_raw": int(raw_rates.shape[1]),
            "n_neurons_modeled": n_neurons,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "solver": solver,
            "checkpoint_every": checkpoint_every,
            "resume_checkpoints": resume_checkpoints,
            "vectorized_christoffel": True,
            "device": str(selected_device),
            "dtype": str(TORCH_DTYPE),
            "filter_train_silent": filter_train_silent,
            "uses_position": False,
            "uses_behavior": False,
            "uses_raw_frame_times": True,
        },
        "geodesic": {
            "parameters": params_geodesic,
            "train": train_geodesic,
            "forecast": forecast_geodesic,
            "AIC": aic_geodesic,
            "BIC": bic_geodesic,
        },
        "free": {
            "parameters": params_free,
            "train": train_free,
            "forecast": forecast_free,
            "AIC": aic_free,
            "BIC": bic_free,
        },
        "preferred_by_AIC": "geodesic" if aic_geodesic < aic_free else "free",
        "preferred_by_BIC": "geodesic" if bic_geodesic < bic_free else "free",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    atomic_torch_save(
        {
            "config": summary["config"],
            "geodesic_state_dict": model_geodesic.state_dict(),
            "free_state_dict": model_free.state_dict(),
            "trial_ids": trial_ids,
            "train_indices": train_indices,
            "heldout_indices": heldout_indices,
            "neuron_mask": neuron_mask,
        },
        output_dir / "model_checkpoint.pt",
    )
    np.savez_compressed(
        output_dir / "fit_outputs.npz",
        trial_ids=trial_ids,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        neuron_mask=neuron_mask,
        lengths=lengths,
        times=arrays["times"],
        valid_mask=valid_mask,
        true_rates=true_rates,
        geodesic_predictions=geodesic_predictions,
        free_predictions=free_predictions,
        geodesic_latents=geodesic_latents,
        free_latents=free_latents,
        geodesic_history_epoch=np.asarray(geodesic_history["epoch"]),
        geodesic_history_nll=np.asarray(geodesic_history["nll"]),
        geodesic_history_mse=np.asarray(geodesic_history["mse"]),
        geodesic_history_r2=np.asarray(geodesic_history["r2"]),
        free_history_epoch=np.asarray(free_history["epoch"]),
        free_history_nll=np.asarray(free_history["nll"]),
        free_history_mse=np.asarray(free_history["mse"]),
        free_history_r2=np.asarray(free_history["r2"]),
        geodesic_context_nll=np.asarray(geodesic_context_history),
        free_context_nll=np.asarray(free_context_history),
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
