#!/usr/bin/env python3
"""Dr-style geodesic v6 model for position-binned E65 trials.

This experiment keeps the successful v6 preprocessing/evaluation path:

    raw x_t -> PCA feature f_t -> reconstructed fhat_t -> raw xhat_t

but replaces the independent neural encoder with per-trial initial conditions
and learned geodesic latent dynamics:

    z_b,0, v_b,0 -> RK4 geodesic steps -> z_b,t -> decoder -> fhat_b,t

Important evaluation caveat:
By default this is "Option B / Dr-style": z0 and v0 are fit for every trial,
including trials that are reported as heldout. The heldout numbers are
therefore reconstruction-fit diagnostics, not strict generalization scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v6_binned_raw_eval_ae as binned_eval  # noqa: E402
import v6_mind_lle as mind  # noqa: E402
import v6_neural_vae as neural  # noqa: E402


def cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def cfg_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg[key.strip()] = mind.parse_value(raw)


def config_lines(cfg: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(k for k in cfg if not str(k).startswith("_")):
        val = cfg[key]
        if isinstance(val, bool):
            sval = "true" if val else "false"
        elif val is None:
            sval = "none"
        else:
            sval = str(val)
        lines.append(f"{key} = {sval}")
    return "\n".join(lines) + "\n"


class MetricNetwork(nn.Module):
    """Positive-definite latent metric g(z) = L(z)L(z)^T + eps I."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32, eps: float = 1e-4):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.eps = float(eps)
        out_dim = self.latent_dim * (self.latent_dim + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        out = self.net(z)
        lmat = z.new_zeros(batch, self.latent_dim, self.latent_dim)
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                if i == j:
                    lmat[:, i, j] = F.softplus(out[:, idx]) + 1e-3
                else:
                    lmat[:, i, j] = out[:, idx]
                idx += 1
        eye = torch.eye(self.latent_dim, dtype=z.dtype, device=z.device).unsqueeze(0)
        return torch.bmm(lmat, lmat.transpose(1, 2)) + self.eps * eye


class GeodesicDynamics(nn.Module):
    """Second-order geodesic dynamics with an optional sensory force."""

    def __init__(
        self,
        metric_net: MetricNetwork,
        sensory_dim: int,
        friction_init: float = -2.0,
    ):
        super().__init__()
        self.metric_net = metric_net
        self.latent_dim = metric_net.latent_dim
        self.sensory_drive = nn.Linear(max(1, int(sensory_dim)), self.latent_dim, bias=False)
        self.log_friction = nn.Parameter(torch.tensor(float(friction_init)))

    def compute_christoffel(self, z: torch.Tensor) -> torch.Tensor:
        z.requires_grad_(True)
        g = self.metric_net(z)
        batch, d, _ = g.shape
        dg = z.new_zeros(batch, d, d, d)
        for i in range(d):
            for j in range(d):
                grad_g = torch.autograd.grad(
                    outputs=g[:, i, j].sum(),
                    inputs=z,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                dg[:, i, j, :] = grad_g

        g_inv = torch.linalg.inv(g)
        gamma = z.new_zeros(batch, d, d, d)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    term = z.new_zeros(batch)
                    for ell in range(d):
                        term = term + 0.5 * g_inv[:, k, ell] * (
                            dg[:, n, ell, m] + dg[:, m, ell, n] - dg[:, m, n, ell]
                        )
                    gamma[:, k, m, n] = term
        return gamma

    def acceleration(self, z: torch.Tensor, v: torch.Tensor, sensory_t: torch.Tensor) -> torch.Tensor:
        gamma = self.compute_christoffel(z)
        geodesic_acc = -torch.einsum("bkmn,bm,bn->bk", gamma, v, v)
        friction = torch.exp(self.log_friction) * v
        return geodesic_acc + self.sensory_drive(sensory_t) - friction

    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        dt: torch.Tensor,
        sensory_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k1_v = self.acceleration(z, v, sensory_t)
        k1_z = v

        k2_v = self.acceleration(z + 0.5 * dt * k1_z, v + 0.5 * dt * k1_v, sensory_t)
        k2_z = v + 0.5 * dt * k1_v

        k3_v = self.acceleration(z + 0.5 * dt * k2_z, v + 0.5 * dt * k2_v, sensory_t)
        k3_z = v + 0.5 * dt * k2_v

        k4_v = self.acceleration(z + dt * k3_z, v + dt * k3_v, sensory_t)
        k4_z = v + dt * k3_v

        v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        z_next = z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        return z_next, v_next


class GeodesicMindAE(nn.Module):
    """Per-trial geodesic latent trajectory plus feature decoder."""

    def __init__(
        self,
        num_trials: int,
        feature_dim: int,
        latent_dim: int,
        sensory_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        metric_hidden: int,
        metric_eps: float,
        init_scale: float,
        friction_init: float,
        log_tau_init: float,
        log_tau_min: float,
        log_tau_max: float,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.log_tau_min = float(log_tau_min)
        self.log_tau_max = float(log_tau_max)
        self.metric = MetricNetwork(latent_dim, metric_hidden, metric_eps)
        self.dynamics = GeodesicDynamics(self.metric, sensory_dim, friction_init)
        self.decoder = self._mlp(latent_dim, feature_dim, hidden, layers, dropout)
        self.z0 = nn.Parameter(torch.randn(num_trials, latent_dim) * float(init_scale))
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim) * float(init_scale))
        self.log_tau = nn.Parameter(torch.tensor(float(log_tau_init)))

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
        blocks: list[nn.Module] = []
        dim = int(in_dim)
        for _ in range(max(1, int(layers))):
            blocks.append(nn.Linear(dim, int(hidden)))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(float(dropout)))
            dim = int(hidden)
        blocks.append(nn.Linear(dim, int(out_dim)))
        return nn.Sequential(*blocks)

    def rollout(self, trial_rows: torch.Tensor, sensory_seq: torch.Tensor, dt_physical: float) -> torch.Tensor:
        z = self.z0[trial_rows]
        v = self.v0[trial_rows]
        log_tau = torch.clamp(self.log_tau, self.log_tau_min, self.log_tau_max)
        dt = torch.as_tensor(float(dt_physical), dtype=z.dtype, device=z.device) * torch.exp(log_tau)
        latents = []
        for t in range(sensory_seq.shape[1]):
            latents.append(z)
            z, v = self.dynamics(z, v, dt, sensory_seq[:, t, :])
        return torch.stack(latents, dim=1)

    def forward(
        self,
        trial_rows: torch.Tensor,
        sensory_seq: torch.Tensor,
        dt_physical: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_seq = self.rollout(trial_rows, sensory_seq, dt_physical)
        flat = z_seq.reshape(-1, z_seq.shape[-1])
        fhat = self.decoder(flat).reshape(z_seq.shape[0], z_seq.shape[1], -1)
        return fhat, z_seq


def build_sensory_sequences(cfg: Dict[str, Any], data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build train/test sensory tensors using the same trial resampling as v6."""

    n_total = int(data["sequences"].shape[0])
    seq_len = int(data["sequences"].shape[1])
    mode = str(cfg.get("geodesic_sensory_mode", "zero")).lower()
    meta: Dict[str, Any] = {"mode": mode}

    if mode in {"zero", "none", "off", "false"}:
        all_sensory = np.zeros((n_total, seq_len, 1), dtype=np.float32)
        meta.update({"keys": [], "normalized": False})
    elif mode in {"axis", "position_axis", "bin_axis"}:
        axis = np.asarray(data["axis"], dtype=np.float32).reshape(1, seq_len, 1)
        all_sensory = np.repeat(axis, n_total, axis=0)
        meta.update({"keys": ["axis"], "normalized": False})
    else:
        if mode in {"evidence", "evidence_smooth", "sensory"}:
            keys = ["EvidenceSmooth"]
        elif mode in {"behavior", "task"}:
            keys = ["EvidenceSmooth", "Velocity", "ChoiceCorrect"]
        else:
            raw_keys = str(cfg.get("geodesic_sensory_keys", mode))
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        npz = np.load(data["data_path"], allow_pickle=True)
        covs = []
        for key in keys:
            if key not in npz.files:
                raise KeyError(f"geodesic sensory key {key!r} not found in {data['data_path']}; keys={npz.files}")
            arr = np.asarray(npz[key], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[0] != len(np.asarray(npz["Trial"]).reshape(-1)):
                raise ValueError(f"Sensory key {key!r} has incompatible shape {arr.shape}")
            covs.append(arr.reshape(arr.shape[0], -1))
        cov = np.concatenate(covs, axis=1).astype(np.float32)
        _, trials, time, position = mind.load_neural_data(data["data_path"])
        cov_seq, cov_trial_ids, _ = mind.build_trial_sequences(cov, trials, time, position, cfg)
        cov_by_trial = {int(tid): cov_seq[i] for i, tid in enumerate(cov_trial_ids)}
        all_sensory = np.stack([cov_by_trial[int(tid)] for tid in data["trial_ids"]], axis=0).astype(np.float32)
        meta.update({"keys": keys, "normalized": cfg_bool(cfg, "geodesic_sensory_normalize", True)})
        if cfg_bool(cfg, "geodesic_sensory_normalize", True):
            train_s = all_sensory[data["train_idx"]].reshape(-1, all_sensory.shape[-1])
            mu = train_s.mean(axis=0, keepdims=True).astype(np.float32)
            sd = train_s.std(axis=0, keepdims=True).astype(np.float32)
            sd = np.where(sd < 1e-6, 1.0, sd)
            all_sensory = ((all_sensory - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)).astype(np.float32)
            meta["mean"] = mu.reshape(-1)
            meta["std"] = sd.reshape(-1)

    scale = cfg_float(cfg, "geodesic_sensory_scale", 1.0)
    all_sensory = (all_sensory * scale).astype(np.float32)
    meta["scale"] = float(scale)
    return all_sensory[data["train_idx"]], all_sensory[data["test_idx"]], meta


def build_fit_arrays(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    teacher: Dict[str, Any],
    sensory_train: np.ndarray,
    sensory_test: np.ndarray,
) -> Dict[str, Any]:
    if data["sequences"] is None:
        raise ValueError("v6_geodesic_binned_raw_eval requires fixed trial sequences, not raw-frame mode.")
    if not cfg_bool(cfg, "geodesic_fit_all_trials", True):
        raise NotImplementedError("This Option B implementation fits z0/v0 for all trials. Set geodesic_fit_all_trials=true.")

    n_train, seq_len, raw_dim = data["x_train_proc"].shape
    n_test = data["x_test_proc"].shape[0]
    feature_dim = data["f_train_flat"].shape[1]

    f_train_seq = data["f_train_flat"].reshape(n_train, seq_len, feature_dim)
    f_test_seq = data["f_test_flat"].reshape(n_test, seq_len, feature_dim)
    z_train_seq = teacher["z_train_norm"].reshape(n_train, seq_len, -1).astype(np.float32)
    z_test_seq = teacher["z_test_norm"].reshape(n_test, seq_len, -1).astype(np.float32)

    teacher_train = teacher["teacher_train_target"].astype(np.float32)
    teacher_test = teacher["teacher_test_target"].astype(np.float32)
    lambda_teacher = cfg_float(cfg, "lambda_teacher_recon", 0.25)
    if teacher_train.shape[1] != feature_dim:
        print("Teacher target is not in PCA feature space; disabling teacher reconstruction loss.")
        teacher_train = data["f_train_flat"].astype(np.float32)
        teacher_test = data["f_test_flat"].astype(np.float32)
        cfg["lambda_teacher_recon"] = 0.0
        lambda_teacher = 0.0
    teacher_train_seq = teacher_train.reshape(n_train, seq_len, feature_dim)
    teacher_test_seq = teacher_test.reshape(n_test, seq_len, feature_dim)

    x_fit = np.concatenate([data["x_train_proc"], data["x_test_proc"]], axis=0).astype(np.float32)
    f_fit = np.concatenate([f_train_seq, f_test_seq], axis=0).astype(np.float32)
    z_teacher_fit = np.concatenate([z_train_seq, z_test_seq], axis=0).astype(np.float32)
    teacher_fit = np.concatenate([teacher_train_seq, teacher_test_seq], axis=0).astype(np.float32)
    sensory_fit = np.concatenate([sensory_train, sensory_test], axis=0).astype(np.float32)
    weights = neural.make_event_weights(x_fit.reshape(-1, raw_dim), cfg).reshape(x_fit.shape[0], seq_len, 1)

    return {
        "x_fit": x_fit,
        "f_fit": f_fit,
        "teacher_fit": teacher_fit,
        "z_teacher_fit": z_teacher_fit,
        "sensory_fit": sensory_fit,
        "weights": weights.astype(np.float32),
        "n_train_trials": int(n_train),
        "n_test_trials": int(n_test),
        "seq_len": int(seq_len),
        "raw_dim": int(raw_dim),
        "feature_dim": int(feature_dim),
        "lambda_teacher_recon_effective": float(lambda_teacher),
    }


def make_model(cfg: Dict[str, Any], fit: Dict[str, Any], device: torch.device) -> GeodesicMindAE:
    return GeodesicMindAE(
        num_trials=fit["x_fit"].shape[0],
        feature_dim=fit["feature_dim"],
        latent_dim=cfg_int(cfg, "latent_dim", 7),
        sensory_dim=fit["sensory_fit"].shape[-1],
        hidden=cfg_int(cfg, "geodesic_decoder_hidden", cfg_int(cfg, "hidden", 256)),
        layers=cfg_int(cfg, "geodesic_decoder_layers", cfg_int(cfg, "layers", 2)),
        dropout=cfg_float(cfg, "dropout", 0.0),
        metric_hidden=cfg_int(cfg, "geodesic_metric_hidden", 32),
        metric_eps=cfg_float(cfg, "geodesic_metric_eps", 1e-4),
        init_scale=cfg_float(cfg, "geodesic_init_scale", 0.1),
        friction_init=cfg_float(cfg, "geodesic_friction_init", -2.0),
        log_tau_init=cfg_float(cfg, "geodesic_log_tau_init", 0.0),
        log_tau_min=cfg_float(cfg, "geodesic_log_tau_min", -5.0),
        log_tau_max=cfg_float(cfg, "geodesic_log_tau_max", 2.0),
    ).to(device)


def train_geodesic_model(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    fit: Dict[str, Any],
    device: torch.device,
) -> tuple[GeodesicMindAE, Dict[str, Any]]:
    model = make_model(cfg, fit, device)

    f_fit = torch.tensor(fit["f_fit"], dtype=torch.float32, device=device)
    x_fit = torch.tensor(fit["x_fit"], dtype=torch.float32, device=device)
    teacher_fit = torch.tensor(fit["teacher_fit"], dtype=torch.float32, device=device)
    z_teacher_fit = torch.tensor(fit["z_teacher_fit"], dtype=torch.float32, device=device)
    sensory_fit = torch.tensor(fit["sensory_fit"], dtype=torch.float32, device=device)
    weights = torch.tensor(fit["weights"], dtype=torch.float32, device=device)

    pca = data["pca"]
    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device) if pca is not None else None
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device) if pca is not None else None
    norm_mu = torch.tensor(data["norm_mu"], dtype=torch.float32, device=device) if data["norm_mu"] is not None else None
    norm_sd = torch.tensor(data["norm_sd"], dtype=torch.float32, device=device) if data["norm_sd"] is not None else None

    lr = cfg_float(cfg, "lr", 1e-3)
    ic_mult = cfg_float(cfg, "geodesic_initial_condition_lr_multiplier", 1.0)
    ic_params = [model.z0, model.v0]
    other_params = [p for name, p in model.named_parameters() if name not in {"z0", "v0"}]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr},
            {"params": ic_params, "lr": lr * ic_mult},
        ],
        weight_decay=cfg_float(cfg, "weight_decay", 1e-5),
    )

    epochs = cfg_int(cfg, "epochs", 100)
    batch_size = min(cfg_int(cfg, "geodesic_trial_batch_size", 8), fit["x_fit"].shape[0])
    rng = np.random.default_rng(cfg_int(cfg, "seed", 42) + 331)
    lambda_feature = cfg_float(cfg, "lambda_feature_recon", 1.0)
    lambda_teacher = cfg_float(cfg, "lambda_teacher_recon", 0.25)
    lambda_z = cfg_float(cfg, "lambda_teacher_z", 0.0)
    lambda_raw = cfg_float(cfg, "lambda_raw_recon", 0.1)
    lambda_latent = cfg_float(cfg, "lambda_geodesic_latent_reg", 0.0)
    lambda_velocity = cfg_float(cfg, "lambda_geodesic_velocity_reg", 0.0)
    dt_physical = cfg_float(cfg, "geodesic_dt", 1.0)
    patience = cfg_int(cfg, "early_stopping_patience", 0)
    min_delta = cfg_float(cfg, "early_stopping_min_delta", 1e-5)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    print(
        "Training GeodesicMindAE: "
        f"fit_trials={fit['x_fit'].shape[0]}, seq_len={fit['seq_len']}, "
        f"feature_dim={fit['feature_dim']}, latent_dim={cfg_int(cfg, 'latent_dim', 7)}, "
        f"sensory_dim={fit['sensory_fit'].shape[-1]}, trial_batch={batch_size}"
    )
    print(
        "Loss weights: "
        f"feature={lambda_feature:g}, raw={lambda_raw:g}, teacher={lambda_teacher:g}, "
        f"teacher_z={lambda_z:g}, latent_reg={lambda_latent:g}, velocity_reg={lambda_velocity:g}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit["x_fit"].shape[0])
        total = 0.0
        parts_sum = {
            "feature": 0.0,
            "raw": 0.0,
            "teacher": 0.0,
            "z": 0.0,
            "latent_reg": 0.0,
            "velocity_reg": 0.0,
        }
        nb = 0
        for start in range(0, order.size, batch_size):
            rows_np = order[start:start + batch_size]
            rows = torch.tensor(rows_np, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            fhat, zseq = model(rows, sensory_fit[rows], dt_physical)
            loss_feature = F.mse_loss(fhat, f_fit[rows])
            loss_teacher = F.mse_loss(fhat, teacher_fit[rows])
            loss_z = F.mse_loss(zseq, z_teacher_fit[rows])
            if lambda_raw > 0:
                xhat = neural.inverse_features_torch(fhat.reshape(-1, fit["feature_dim"]), pca_components, pca_mean, norm_mu, norm_sd)
                xhat = xhat.reshape_as(x_fit[rows])
                loss_raw = torch.mean(weights[rows] * (xhat - x_fit[rows]) ** 2)
            else:
                loss_raw = torch.zeros((), dtype=torch.float32, device=device)
            loss_latent = torch.mean(zseq ** 2)
            loss_velocity = torch.mean(model.v0[rows] ** 2)
            loss = (
                lambda_feature * loss_feature
                + lambda_teacher * loss_teacher
                + lambda_z * loss_z
                + lambda_raw * loss_raw
                + lambda_latent * loss_latent
                + lambda_velocity * loss_velocity
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite geodesic loss at epoch {epoch}, batch starting {start}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_float(cfg, "grad_clip", 5.0))
            opt.step()

            total += float(loss.detach().cpu())
            parts_sum["feature"] += float(loss_feature.detach().cpu())
            parts_sum["raw"] += float(loss_raw.detach().cpu())
            parts_sum["teacher"] += float(loss_teacher.detach().cpu())
            parts_sum["z"] += float(loss_z.detach().cpu())
            parts_sum["latent_reg"] += float(loss_latent.detach().cpu())
            parts_sum["velocity_reg"] += float(loss_velocity.detach().cpu())
            nb += 1

        avg = total / max(1, nb)
        parts_avg = {k: v / max(1, nb) for k, v in parts_sum.items()}
        row = {
            "epoch": int(epoch),
            "loss": float(avg),
            "log_tau": float(model.log_tau.detach().cpu()),
            "friction": float(torch.exp(model.dynamics.log_friction.detach()).cpu()),
            **parts_avg,
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, cfg_int(cfg, "log_every", 10)) == 0:
            print(
                f"[{epoch:03d}] loss {avg:.5f} | feature {parts_avg['feature']:.5f} | "
                f"raw {parts_avg['raw']:.5f} | teacher {parts_avg['teacher']:.5f} | "
                f"z {parts_avg['z']:.5f} | log_tau {row['log_tau']:.3f}"
            )

        if avg < best_loss - min_delta:
            best_loss = avg
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} loss {best_loss:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "fit_all_trials": cfg_bool(cfg, "geodesic_fit_all_trials", True),
        "strict_heldout": False,
    }


def predict_geodesic_features(
    model: GeodesicMindAE,
    sensory: np.ndarray,
    device: torch.device,
    batch_size: int,
    dt_physical: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    params = list(model.parameters())
    requires = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    f_chunks = []
    z_chunks = []
    try:
        for start in range(0, sensory.shape[0], batch_size):
            rows_np = np.arange(start, min(start + batch_size, sensory.shape[0]), dtype=np.int64)
            rows = torch.tensor(rows_np, dtype=torch.long, device=device)
            sensory_t = torch.tensor(sensory[rows_np], dtype=torch.float32, device=device)
            with torch.enable_grad():
                fhat, zseq = model(rows, sensory_t, dt_physical)
            f_chunks.append(fhat.detach().cpu().numpy())
            z_chunks.append(zseq.detach().cpu().numpy())
    finally:
        for p, req in zip(params, requires):
            p.requires_grad_(req)
    return np.concatenate(f_chunks, axis=0).astype(np.float32), np.concatenate(z_chunks, axis=0).astype(np.float32)


def split_predictions(
    fhat_fit: np.ndarray,
    z_fit: np.ndarray,
    fit: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = fit["n_train_trials"]
    return fhat_fit[:n_train], fhat_fit[n_train:], z_fit[:n_train], z_fit[n_train:]


def evaluate_and_save(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    data: Dict[str, Any],
    teacher: Dict[str, Any],
    fit: Dict[str, Any],
    sensory_meta: Dict[str, Any],
    model: GeodesicMindAE,
    train_meta: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    pred_batch = cfg_int(cfg, "geodesic_predict_trial_batch_size", cfg_int(cfg, "geodesic_trial_batch_size", 8))
    fhat_fit, z_fit = predict_geodesic_features(
        model,
        fit["sensory_fit"],
        device,
        pred_batch,
        cfg_float(cfg, "geodesic_dt", 1.0),
    )
    fhat_train, fhat_test, zhat_train, zhat_test = split_predictions(fhat_fit, z_fit, fit)
    x_train_shape = data["x_train_proc"].shape
    x_test_shape = data["x_test_proc"].shape
    xhat_train_flat = neural.inverse_features_numpy(
        fhat_train.reshape(-1, fit["feature_dim"]),
        data["pca"],
        data["norm_mu"],
        data["norm_sd"],
    )
    xhat_test_flat = neural.inverse_features_numpy(
        fhat_test.reshape(-1, fit["feature_dim"]),
        data["pca"],
        data["norm_mu"],
        data["norm_sd"],
    )
    xhat_train_flat, xhat_test_flat, postprocess_meta = mind.apply_reconstruction_postprocess(
        xhat_train_flat,
        xhat_test_flat,
        data["x_train_flat"],
        cfg,
    )
    xhat_train = xhat_train_flat.reshape(x_train_shape)
    xhat_test = xhat_test_flat.reshape(x_test_shape)

    binned_train_metrics = mind.corr_and_r2(data["x_train_proc"], xhat_train)
    binned_train_metrics.update(mind.summarize_trial_mind_r2(data["x_train_proc"], xhat_train))
    binned_test_metrics = mind.corr_and_r2(data["x_test_proc"], xhat_test)
    binned_test_metrics.update(mind.summarize_trial_mind_r2(data["x_test_proc"], xhat_test))
    binned_event_metrics = mind.compute_event_metrics(
        data["x_test_proc"],
        xhat_test,
        percentile=cfg_float(cfg, "event_metric_percentile", 99.0),
    )

    raw_eval = binned_eval.build_raw_frame_split_for_binned_trials(cfg, data)
    train_trial_ids = data["trial_ids"][data["train_idx"]]
    test_trial_ids = data["trial_ids"][data["test_idx"]]
    bin_axis = np.asarray(data["axis"], dtype=np.float64)
    xhat_train_raw_flat = binned_eval.map_binned_predictions_to_raw_frames(
        xhat_train,
        train_trial_ids,
        raw_eval["train_frame_trial_ids"],
        raw_eval["train_frame_axis"],
        bin_axis,
    )
    xhat_test_raw_flat = binned_eval.map_binned_predictions_to_raw_frames(
        xhat_test,
        test_trial_ids,
        raw_eval["test_frame_trial_ids"],
        raw_eval["test_frame_axis"],
        bin_axis,
    )
    xhat_train_raw = xhat_train_raw_flat[None, :, :]
    xhat_test_raw = xhat_test_raw_flat[None, :, :]

    raw_train_metrics = mind.corr_and_r2(raw_eval["train_x"], xhat_train_raw)
    raw_train_metrics.update(mind.summarize_frame_trial_mind_r2(
        raw_eval["train_x"],
        xhat_train_raw,
        raw_eval["train_frame_trial_ids"],
        raw_eval["train_trial_ids"],
    ))
    raw_test_metrics = mind.corr_and_r2(raw_eval["test_x"], xhat_test_raw)
    raw_test_metrics.update(mind.summarize_frame_trial_mind_r2(
        raw_eval["test_x"],
        xhat_test_raw,
        raw_eval["test_frame_trial_ids"],
        raw_eval["test_trial_ids"],
    ))
    raw_event_metrics = mind.compute_event_metrics(
        raw_eval["test_x"],
        xhat_test_raw,
        percentile=cfg_float(cfg, "event_metric_percentile", 99.0),
    )

    print(
        f"Binned heldout-fit r {binned_test_metrics['r']:.4f} | R2 {binned_test_metrics['R2']:.4f} | "
        f"MIND_R2 {binned_test_metrics['MIND_R2']:.4f} | trial median {binned_test_metrics['MIND_R2_trial_median']:.4f}"
    )
    print(
        f"Raw mapped heldout-fit r {raw_test_metrics['r']:.4f} | R2 {raw_test_metrics['R2']:.4f} | "
        f"MIND_R2 {raw_test_metrics['MIND_R2']:.4f} | trial median {raw_test_metrics['MIND_R2_trial_median']:.4f}"
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": mind.jsonable(cfg),
            "train_meta": mind.jsonable(train_meta),
            "sensory_meta": mind.jsonable(sensory_meta),
        },
        out_dir / "model.pt",
    )
    np.savez_compressed(
        out_dir / "analysis_cache_best.npz",
        x_true=raw_eval["test_x"].astype(np.float32),
        x_pred=xhat_test_raw.astype(np.float32),
        x_binned_true=data["x_test_proc"].astype(np.float32),
        x_binned_pred=xhat_test.astype(np.float32),
        x_train_true=raw_eval["train_x"].astype(np.float32),
        x_train_pred=xhat_train_raw.astype(np.float32),
        z_test=zhat_test.astype(np.float32),
        z_train=zhat_train.astype(np.float32),
        train_trial_ids=train_trial_ids,
        test_trial_ids=test_trial_ids,
        train_frame_trial_ids=raw_eval["train_frame_trial_ids"],
        test_frame_trial_ids=raw_eval["test_frame_trial_ids"],
        train_frame_time=raw_eval["train_frame_time"],
        test_frame_time=raw_eval["test_frame_time"],
        train_frame_position=raw_eval["train_frame_position"],
        test_frame_position=raw_eval["test_frame_position"],
        train_frame_axis=raw_eval["train_frame_axis"],
        test_frame_axis=raw_eval["test_frame_axis"],
        metrics_json=np.asarray(json.dumps(mind.jsonable({"binned": binned_test_metrics, "raw_mapped": raw_test_metrics}))),
    )
    binned_eval.save_raw_mapped_heatmap(
        raw_eval["test_x"],
        xhat_test_raw,
        out_dir / "raw_vs_recon_t0_15_n0_40.png",
        "Geodesic v6 reconstruction mapped to raw frames",
        raw_eval["test_frame_trial_ids"],
        raw_eval["test_frame_time"],
        trial_index=cfg_int(cfg, "raw_eval_plot_trial_index", 0),
        frame_start=cfg_int(cfg, "raw_eval_plot_frame_start", 0),
        frame_end=cfg_int(cfg, "raw_eval_plot_frame_end", 15),
        neuron_start=cfg_int(cfg, "raw_eval_plot_neuron_start", 0),
        neuron_end=cfg_int(cfg, "raw_eval_plot_neuron_end", 40),
    )
    mind.save_recon_heatmap(
        data["x_test_proc"],
        xhat_test,
        out_dir / "binned_vs_recon_t0_15_n0_40.png",
        "Geodesic v6 heldout-fit binned reconstruction",
    )
    mind.save_embedding_plot(teacher["z_lm"], out_dir / "latent_manifold_mds.png")

    final_metrics = {
        "binned_train": binned_train_metrics,
        "binned_heldout_fit": binned_test_metrics,
        "binned_events": binned_event_metrics,
        "raw_mapped_train": raw_train_metrics,
        "raw_mapped_heldout_fit": raw_test_metrics,
        "raw_mapped_events": raw_event_metrics,
        "R2": raw_test_metrics["R2"],
        "r": raw_test_metrics["r"],
        "strict_heldout": False,
    }
    torch.save(final_metrics, out_dir / "final_metrics.pt")
    metadata = {
        "script": Path(__file__).name,
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(data["data_path"]),
        "model_type": "geodesic_v6_dr_style",
        "initial_conditions": "fit_per_trial_z0_v0_for_train_and_heldout",
        "strict_heldout": False,
        "data_mode": "position_binned_train_raw_frame_eval",
        "train_alignment": str(cfg.get("sequence_alignment", "position")),
        "sequence_length": int(data["x_train_proc"].shape[1]),
        "n_trials_total": int(data["sequences"].shape[0]),
        "n_trials_train": int(len(data["train_idx"])),
        "n_trials_heldout": int(len(data["test_idx"])),
        "heldout_fraction": float(len(data["test_idx"]) / data["sequences"].shape[0]),
        "n_train_binned_states": int(data["x_train_flat"].shape[0]),
        "n_heldout_binned_states": int(data["x_test_flat"].shape[0]),
        "n_train_raw_frames": int(raw_eval["n_train_frames"]),
        "n_heldout_raw_frames": int(raw_eval["n_test_frames"]),
        "n_neurons": int(data["x_train_proc"].shape[-1]),
        "feature_dim": int(data["f_train_flat"].shape[1]),
        "latent_dim": cfg_int(cfg, "latent_dim", 7),
        "pca_explained_variance": data["pca_var"],
        "sensory": sensory_meta,
        "postprocess_meta": postprocess_meta,
        "raw_eval": {k: v for k, v in raw_eval.items() if not isinstance(v, np.ndarray)},
        "transition_distance_meta": teacher["dist_meta"],
        "embedding_meta": teacher["embed_meta"],
        "landmark_selection": teacher["landmark_meta"],
        "manifold_filter": teacher["manifold_meta"],
        "training": train_meta,
        "metrics": final_metrics,
        "method_note": (
            "Dr-style geodesic v6: each trial has fitted latent z0/v0; the shared metric, "
            "time scale, friction, sensory force, and decoder are fit across all trials. "
            "Because heldout trials also have fitted initial conditions, heldout-fit "
            "metrics are reconstruction diagnostics rather than strict trial-generalization metrics."
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(mind.jsonable(metadata), indent=2) + "\n")
    (out_dir / "training_results.txt").write_text(json.dumps(mind.jsonable(final_metrics), indent=2) + "\n")
    (out_dir / "config.txt").write_text(config_lines(cfg))
    print(f"Saved outputs to {out_dir}")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v6_geodesic_binned_raw_eval.txt")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config key=value")
    args = parser.parse_args()

    cfg = mind.load_config(args.config)
    apply_overrides(cfg, args.overrides)
    seed = cfg_int(cfg, "seed", cfg_int(cfg, "mind_split_seed", 42))
    neural.set_torch_seed(seed)
    device = neural.device_from_config(cfg)
    print(f"Device: {device}")

    out_dir = Path(cfg.get("out_dir", "runs/v6_geodesic_binned_raw_eval_manual")).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = neural.prepare_data(cfg, seed)
    teacher = neural.build_mind_teacher(cfg, data)
    sensory_train, sensory_test, sensory_meta = build_sensory_sequences(cfg, data)
    fit = build_fit_arrays(cfg, data, teacher, sensory_train, sensory_test)
    model, train_meta = train_geodesic_model(cfg, data, fit, device)
    evaluate_and_save(cfg, args, out_dir, data, teacher, fit, sensory_meta, model, train_meta, device)


if __name__ == "__main__":
    main()
