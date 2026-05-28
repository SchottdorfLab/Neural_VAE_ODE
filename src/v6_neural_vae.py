#!/usr/bin/env python3
"""MIND-distilled neural autoencoder for E65 population activity.

This file intentionally replaces the old v6 ODE/VAE experiment with a simpler
neural version of the successful `v6_mind_lle.py` pipeline.

Core assumptions carried over from the MIND-LLE scan:
- Neural population states lie near a low-dimensional manifold.
- PCA features are used for geometry/reconstruction stability, then mapped back
  to raw neuron activity for metrics and plots.
- The useful geometry is local graph/geodesic geometry, not global Euclidean
  geometry.
- The currently best held-out setting is static local geometry:
  transition_probability_mode=local and transition_weight=0.
- No VAE sampling and no latent ODE dynamics. This is a deterministic AE.

Training target:
1. Build the same train-only MIND-LLE geometry as `v6_mind_lle.py`.
2. Use LLE to map every train state to teacher manifold coordinates z_MIND.
3. Train an MLP encoder/decoder: PCA feature -> z -> PCA feature.
4. Optionally distill the nonparametric MIND decoder's reconstruction.
5. Evaluate held-out trials only after all preprocessing/geometry is fit on train.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v6_mind_lle as mind  # noqa: E402


def cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def cfg_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


class MindDistilledAE(nn.Module):
    """Deterministic feature -> manifold -> feature autoencoder."""

    def __init__(self, feature_dim: int, latent_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = self._mlp(feature_dim, latent_dim, hidden, layers, dropout)
        self.decoder = self._mlp(latent_dim, feature_dim, hidden, layers, dropout)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
        layers = max(1, int(layers))
        blocks: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(dim, hidden))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            dim = hidden
        blocks.append(nn.Linear(dim, out_dim))
        return nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(features)
        fhat = self.decoder(z)
        return fhat, z


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "auto")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_trials(x: np.ndarray) -> np.ndarray:
    return mind.flatten_trials(x)


def inverse_features_numpy(
    fhat: np.ndarray,
    pca: PCA | None,
    norm_mu: np.ndarray | None,
    norm_sd: np.ndarray | None,
) -> np.ndarray:
    xhat = pca.inverse_transform(fhat).astype(np.float32) if pca is not None else fhat.astype(np.float32)
    if norm_mu is not None and norm_sd is not None:
        xhat = (xhat * norm_sd + norm_mu).astype(np.float32)
    return xhat


def inverse_features_torch(
    fhat: torch.Tensor,
    pca_components: torch.Tensor | None,
    pca_mean: torch.Tensor | None,
    norm_mu: torch.Tensor | None,
    norm_sd: torch.Tensor | None,
) -> torch.Tensor:
    if pca_components is not None and pca_mean is not None:
        xhat = fhat @ pca_components + pca_mean
    else:
        xhat = fhat
    if norm_mu is not None and norm_sd is not None:
        xhat = xhat * norm_sd + norm_mu
    return xhat


def make_event_weights(x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    alpha = cfg_float(cfg, "event_weight_alpha", 0.0)
    if alpha <= 0:
        return np.ones((x.shape[0], 1), dtype=np.float32)
    mode = str(cfg.get("event_weight_mode", "activity_or_dx")).lower()
    activity = np.max(np.maximum(x, 0.0), axis=1)
    dx = np.zeros_like(x)
    seq_len = cfg_int(cfg, "_seq_len", 1)
    if seq_len > 1 and x.shape[0] % seq_len == 0:
        xx = x.reshape(-1, seq_len, x.shape[1])
        dd = mind.sequence_deltas(xx).reshape(x.shape)
        dx = np.abs(dd)
    dyn = np.max(dx, axis=1)
    if mode == "activity":
        score = activity
    elif mode in {"dx", "dynamics", "transition"}:
        score = dyn
    else:
        score = np.maximum(activity, dyn)
    pct = cfg_float(cfg, "event_weight_percentile", 95.0)
    scale = np.percentile(score, pct)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.max(score) if np.max(score) > 1e-8 else 1.0
    score = np.clip(score / scale, 0.0, 10.0)
    return (1.0 + alpha * score[:, None]).astype(np.float32)


def prepare_data(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = mind.set_seed(seed)
    data_path = mind.resolve_path(cfg.get("data_path", "src/npz_e65_data/E65_data.npz"))
    print(f"Loading data: {data_path}")
    roi, trials, time, position = mind.load_neural_data(data_path)
    print(f"Raw ROI: frames={roi.shape[0]}, neurons={roi.shape[1]}")

    if cfg_bool(cfg, "filter_inactive_neurons", True):
        mask = mind.mind_active_neuron_mask(roi, cfg_float(cfg, "inactive_neuron_threshold", 0.0))
        removed = int((~mask).sum())
        roi = roi[:, mask]
        print(f"MIND neuron filter: removed {removed} globally silent neurons; kept {roi.shape[1]} neurons.")
    else:
        mask = np.ones(roi.shape[1], dtype=bool)
        removed = 0

    sequences, trial_ids, axis = mind.build_trial_sequences(roi, trials, time, position, cfg)
    print(f"Built sequences: B={sequences.shape[0]}, L={sequences.shape[1]}, N={sequences.shape[2]}")
    cfg["_seq_len"] = int(sequences.shape[1])

    if cfg_bool(cfg, "mind_style_eval_all", False):
        train_idx = np.arange(sequences.shape[0], dtype=np.int64)
        test_idx = np.arange(sequences.shape[0], dtype=np.int64)
        print(f"MIND-style all-points eval: fit={len(train_idx)}, eval={len(test_idx)}")
    else:
        train_idx, test_idx = mind.split_trials(
            sequences.shape[0],
            cfg_float(cfg, "test_frac", cfg_float(cfg, "mind_test_frac", 0.1)),
            cfg_int(cfg, "mind_split_seed", seed),
        )
        print(f"Trial split: train={len(train_idx)}, heldout={len(test_idx)} ({len(test_idx) / sequences.shape[0]:.3f})")

    x_train = sequences[train_idx]
    x_test = sequences[test_idx]

    if cfg_bool(cfg, "filter_train_silent_neurons", False):
        std = flatten_trials(x_train).std(axis=0)
        keep = std > cfg_float(cfg, "train_silent_threshold", cfg_float(cfg, "train_silent_std_threshold", 1e-7))
        removed_train = int((~keep).sum())
        x_train = x_train[..., keep]
        x_test = x_test[..., keep]
        mask_indices = np.flatnonzero(mask)
        mask[mask_indices[~keep]] = False
        print(f"Train-std neuron filter: removed {removed_train}; kept {x_train.shape[-1]} neurons.")

    if cfg_bool(cfg, "baseline_correct", False):
        q = cfg_float(cfg, "baseline_quantile", 0.1)
        x_train_proc = mind.baseline_correct_trials(x_train, q)
        x_test_proc = mind.baseline_correct_trials(x_test, q)
    else:
        x_train_proc = x_train.astype(np.float32)
        x_test_proc = x_test.astype(np.float32)

    x_train_flat = flatten_trials(x_train_proc)
    x_test_flat = flatten_trials(x_test_proc)

    norm_mu = None
    norm_sd = None
    if cfg_bool(cfg, "normalize_inputs", False):
        norm_mu = x_train_flat.mean(axis=0, keepdims=True).astype(np.float32)
        norm_sd = x_train_flat.std(axis=0, keepdims=True).astype(np.float32)
        norm_sd = np.where(norm_sd < cfg_float(cfg, "normalize_eps", 1e-6), 1.0, norm_sd)
        x_train_flat_proc = ((x_train_flat - norm_mu) / norm_sd).astype(np.float32)
        x_test_flat_proc = ((x_test_flat - norm_mu) / norm_sd).astype(np.float32)
    else:
        x_train_flat_proc = x_train_flat.astype(np.float32)
        x_test_flat_proc = x_test_flat.astype(np.float32)

    use_pca = cfg_bool(cfg, "use_pca", True) or cfg_int(cfg, "pca_dim", 0) > 0
    pca = None
    if use_pca:
        pca_dim = cfg_int(cfg, "pca_dim", 50)
        if pca_dim <= 0:
            pca_dim = min(x_train_flat_proc.shape[1], cfg_int(cfg, "pca_max_dim", 64))
        pca = PCA(n_components=min(pca_dim, x_train_flat_proc.shape[1]), random_state=seed)
        f_train_flat = pca.fit_transform(x_train_flat_proc).astype(np.float32)
        f_test_flat = pca.transform(x_test_flat_proc).astype(np.float32)
        pca_var = float(np.sum(pca.explained_variance_ratio_))
        print(f"PCA feature space: dim={f_train_flat.shape[1]}, explained_variance={pca_var:.4f}")
    else:
        f_train_flat = x_train_flat_proc.astype(np.float32)
        f_test_flat = x_test_flat_proc.astype(np.float32)
        pca_var = float("nan")
        print(f"Feature space: raw neurons, dim={f_train_flat.shape[1]}")

    return {
        "rng": rng,
        "data_path": data_path,
        "mask": mask,
        "globally_silent_removed": removed,
        "sequences": sequences,
        "trial_ids": trial_ids,
        "axis": axis,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "x_train_proc": x_train_proc,
        "x_test_proc": x_test_proc,
        "x_train_flat": x_train_flat,
        "x_test_flat": x_test_flat,
        "f_train_flat": f_train_flat,
        "f_test_flat": f_test_flat,
        "pca": pca,
        "pca_var": pca_var,
        "norm_mu": norm_mu,
        "norm_sd": norm_sd,
    }


def build_mind_teacher(cfg: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    rng = data["rng"]
    f_train_flat = data["f_train_flat"]
    x_train_flat = data["x_train_flat"]
    seq_len = data["x_train_proc"].shape[1]
    feat_dim = f_train_flat.shape[1]
    f_train_seq = f_train_flat.reshape(data["x_train_proc"].shape[0], seq_len, feat_dim)
    delta_train_flat = flatten_trials(mind.sequence_deltas(f_train_seq))
    raw_delta_train_flat = flatten_trials(mind.sequence_deltas(data["x_train_proc"]))

    manifold_mask = np.ones(f_train_flat.shape[0], dtype=bool)
    manifold_meta: Dict[str, Any] = {
        "filter_silent_frames_for_manifold": cfg_bool(cfg, "filter_silent_frames_for_manifold", False),
        "total_train_states": int(f_train_flat.shape[0]),
    }
    if cfg_bool(cfg, "filter_silent_frames_for_manifold", False):
        score_mode = str(cfg.get("silent_frame_score", "sum"))
        threshold = cfg_float(cfg, "silent_frame_threshold", 0.0)
        frame_score = mind.score_activity(x_train_flat, score_mode)
        manifold_mask = frame_score > threshold
        manifold_meta.update({
            "silent_frame_score": score_mode,
            "silent_frame_threshold": threshold,
            "active_train_states": int(manifold_mask.sum()),
            "removed_silent_train_states": int((~manifold_mask).sum()),
            "active_fraction": float(manifold_mask.mean()),
        })
        print(
            "Manifold active-frame filter: "
            f"kept {manifold_meta['active_train_states']} / {manifold_meta['total_train_states']} "
            f"train states ({manifold_meta['active_fraction']:.3f})"
        )

    manifold_indices = np.flatnonzero(manifold_mask)
    landmark_count = cfg_int(cfg, "landmark_count", min(750, f_train_flat.shape[0]))
    lm_local, landmark_meta = mind.select_landmarks(
        f_train_flat[manifold_mask],
        x_train_flat[manifold_mask],
        raw_delta_train_flat[manifold_mask],
        landmark_count,
        rng,
        cfg,
    )
    lm_idx = manifold_indices[lm_local]
    f_lm = f_train_flat[lm_idx]
    df_lm = delta_train_flat[lm_idx]
    print(
        f"Selected train landmarks: {f_lm.shape[0]} of {f_train_flat.shape[0]} states "
        f"({landmark_meta['mode']}: activity={landmark_meta['activity_count']}, "
        f"transition={landmark_meta['transition_count']}, coverage={landmark_meta['coverage_count']})"
    )

    d_mind, dist_meta = mind.transition_geodesic_distances(
        f_lm,
        df_lm,
        k=cfg_int(cfg, "transition_k", 16),
        transition_weight=cfg_float(cfg, "transition_weight", 0.0),
        temperature=cfg_float(cfg, "transition_temperature", 1.0),
        sym=str(cfg.get("graph_sym", "min")),
        probability_mode=str(cfg.get("transition_probability_mode", "local")),
        exclude_self=cfg_bool(cfg, "transition_exclude_self", True),
        use_graph_geodesics=cfg_bool(cfg, "use_graph_geodesics", True),
    )
    print(f"Built MIND distances: max={dist_meta['distance_max']:.4f}, disconnected_replaced={dist_meta['disconnected_pairs_replaced']}")

    latent_dim = cfg_int(cfg, "latent_dim", 7)
    z_lm, embed_meta = mind.embed_distances(d_mind, latent_dim, cfg)
    print(f"Embedded landmarks: Z={z_lm.shape}, mode={embed_meta['embedding_mode']}")

    lle_k = cfg_int(cfg, "lle_k", 16)
    lle_ridge = cfg_float(cfg, "lle_ridge", 0.001)
    f2z = mind.LLEMapper(lle_k, lle_ridge).fit(f_lm, z_lm)
    z_train = f2z.transform(data["f_train_flat"])
    z_test = f2z.transform(data["f_test_flat"])

    z_mu = z_train.mean(axis=0, keepdims=True).astype(np.float32)
    z_sd = z_train.std(axis=0, keepdims=True).astype(np.float32)
    z_sd = np.where(z_sd < 1e-6, 1.0, z_sd)
    z_train_norm = ((z_train - z_mu) / z_sd).astype(np.float32)
    z_test_norm = ((z_test - z_mu) / z_sd).astype(np.float32)

    inverse_decoder = str(cfg.get("inverse_decoder", "kernel")).lower()
    inverse_k = cfg_int(cfg, "inverse_k", cfg_int(cfg, "kernel_k", lle_k))
    kernel_bandwidth = cfg_float(cfg, "kernel_bandwidth", 0.35)
    decoder_target = str(cfg.get("decoder_target", "feature")).lower()
    if decoder_target in {"raw", "neural", "activity"}:
        decoder_train_target = data["x_train_flat"]
    else:
        decoder_train_target = data["f_train_flat"]
    if inverse_decoder in {"kernel", "rbf", "weighted", "weighted_average"}:
        z2target = mind.KernelRegressor(inverse_k, kernel_bandwidth).fit(z_train, decoder_train_target)
    elif inverse_decoder in {"lle", "local_linear"}:
        z2target = mind.LLEMapper(inverse_k, lle_ridge).fit(z_train, decoder_train_target)
    else:
        raise ValueError(f"Unknown inverse_decoder={inverse_decoder!r}")
    teacher_train_target = z2target.transform(z_train).astype(np.float32)
    teacher_test_target = z2target.transform(z_test).astype(np.float32)

    return {
        "lm_idx": lm_idx,
        "f_lm": f_lm,
        "z_lm": z_lm,
        "d_mind": d_mind,
        "z_train": z_train.astype(np.float32),
        "z_test": z_test.astype(np.float32),
        "z_train_norm": z_train_norm,
        "z_test_norm": z_test_norm,
        "z_mu": z_mu,
        "z_sd": z_sd,
        "teacher_train_target": teacher_train_target,
        "teacher_test_target": teacher_test_target,
        "decoder_target": decoder_target,
        "dist_meta": dist_meta,
        "embed_meta": embed_meta,
        "landmark_meta": landmark_meta,
        "manifold_meta": manifold_meta,
    }


def train_model(cfg: Dict[str, Any], data: Dict[str, Any], teacher: Dict[str, Any], device: torch.device) -> Tuple[MindDistilledAE, Dict[str, Any]]:
    f_train = data["f_train_flat"].astype(np.float32)
    z_teacher = teacher["z_train_norm"].astype(np.float32)
    teacher_target = teacher["teacher_train_target"].astype(np.float32)
    if teacher_target.shape[1] != f_train.shape[1]:
        print(
            "Teacher reconstruction target is not in PCA feature space; "
            "disabling teacher reconstruction loss for the neural feature decoder."
        )
        teacher_target = f_train
        cfg["lambda_teacher_recon"] = 0.0
    x_train = data["x_train_flat"].astype(np.float32)
    weights = make_event_weights(x_train, cfg)

    n = f_train.shape[0]
    val_frac = cfg_float(cfg, "state_val_frac", 0.1)
    rng = np.random.default_rng(cfg_int(cfg, "seed", 42) + 17)
    order = rng.permutation(n)
    n_val = int(round(n * val_frac)) if val_frac > 0 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n_val > 0 else order

    tensors = {
        "f": torch.from_numpy(f_train),
        "z": torch.from_numpy(z_teacher),
        "teacher": torch.from_numpy(teacher_target),
        "x": torch.from_numpy(x_train),
        "w": torch.from_numpy(weights),
    }

    def subset(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(*(t[indices] for t in tensors.values()))

    batch_size = cfg_int(cfg, "batch_size", 512)
    train_loader = DataLoader(subset(train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(subset(val_idx), batch_size=batch_size, shuffle=False, drop_last=False) if n_val > 0 else None

    feature_dim = f_train.shape[1]
    latent_dim = cfg_int(cfg, "latent_dim", 7)
    model = MindDistilledAE(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden=cfg_int(cfg, "hidden", 256),
        layers=cfg_int(cfg, "layers", cfg_int(cfg, "encoder_layers", 3)),
        dropout=cfg_float(cfg, "dropout", 0.0),
    ).to(device)

    pca = data["pca"]
    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device) if pca is not None else None
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device) if pca is not None else None
    norm_mu = torch.tensor(data["norm_mu"], dtype=torch.float32, device=device) if data["norm_mu"] is not None else None
    norm_sd = torch.tensor(data["norm_sd"], dtype=torch.float32, device=device) if data["norm_sd"] is not None else None

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_float(cfg, "lr", 1e-3),
        weight_decay=cfg_float(cfg, "weight_decay", 1e-5),
    )
    epochs = cfg_int(cfg, "epochs", 300)
    patience = cfg_int(cfg, "early_stopping_patience", 50)
    min_delta = cfg_float(cfg, "early_stopping_min_delta", 1e-5)
    lambda_feature = cfg_float(cfg, "lambda_feature_recon", 1.0)
    lambda_z = cfg_float(cfg, "lambda_teacher_z", 1.0)
    lambda_teacher = cfg_float(cfg, "lambda_teacher_recon", 0.25)
    lambda_raw = cfg_float(cfg, "lambda_raw_recon", 0.0)

    def loss_on_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, Dict[str, float]]:
        f, zt, ft, x, w = (b.to(device) for b in batch)
        fhat, zhat = model(f)
        loss_feature = F.mse_loss(fhat, f)
        loss_z = F.mse_loss(zhat, zt)
        loss_teacher = F.mse_loss(fhat, ft)
        if lambda_raw > 0:
            xhat = inverse_features_torch(fhat, pca_components, pca_mean, norm_mu, norm_sd)
            loss_raw = torch.mean(w * (xhat - x) ** 2)
        else:
            loss_raw = torch.zeros((), device=device)
        loss = lambda_feature * loss_feature + lambda_z * loss_z + lambda_teacher * loss_teacher + lambda_raw * loss_raw
        return loss, {
            "feature": float(loss_feature.detach().cpu()),
            "z": float(loss_z.detach().cpu()),
            "teacher": float(loss_teacher.detach().cpu()),
            "raw": float(loss_raw.detach().cpu()),
        }

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        nb = 0
        parts_sum = {"feature": 0.0, "z": 0.0, "teacher": 0.0, "raw": 0.0}
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_on_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_float(cfg, "grad_clip", 5.0))
            opt.step()
            train_loss += float(loss.detach().cpu())
            for k, v in parts.items():
                parts_sum[k] += v
            nb += 1
        train_loss /= max(1, nb)
        parts_avg = {k: v / max(1, nb) for k, v in parts_sum.items()}

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            vb = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss, _ = loss_on_batch(batch)
                    val_loss += float(loss.detach().cpu())
                    vb += 1
            val_loss /= max(1, vb)
        else:
            val_loss = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **parts_avg})
        if epoch == 1 or epoch % max(1, cfg_int(cfg, "log_every", 25)) == 0:
            print(
                f"[{epoch:03d}] train {train_loss:.5f} | val {val_loss:.5f} | "
                f"feature {parts_avg['feature']:.5f} | z {parts_avg['z']:.5f} | teacher {parts_avg['teacher']:.5f} | raw {parts_avg['raw']:.5f}"
            )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} val {best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val}


def predict_features(model: MindDistilledAE, features: np.ndarray, device: torch.device, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    fhat_chunks = []
    z_chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start:start + batch_size].astype(np.float32)).to(device)
            fhat, z = model(batch)
            fhat_chunks.append(fhat.cpu().numpy())
            z_chunks.append(z.cpu().numpy())
    return np.concatenate(fhat_chunks, axis=0).astype(np.float32), np.concatenate(z_chunks, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v6_mind_geometry.txt")
    args = parser.parse_args()

    cfg = mind.load_config(args.config)
    seed = cfg_int(cfg, "seed", cfg_int(cfg, "mind_split_seed", 42))
    set_torch_seed(seed)
    device = device_from_config(cfg)
    print(f"Device: {device}")

    out_dir = Path(cfg.get("out_dir", "runs/v6_neural_mind_ae_manual")).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(cfg, seed)
    teacher = build_mind_teacher(cfg, data)

    model, train_meta = train_model(cfg, data, teacher, device)

    fhat_train, zhat_train = predict_features(model, data["f_train_flat"], device, cfg_int(cfg, "predict_batch_size", 4096))
    fhat_test, zhat_test = predict_features(model, data["f_test_flat"], device, cfg_int(cfg, "predict_batch_size", 4096))

    xhat_train_flat = inverse_features_numpy(fhat_train, data["pca"], data["norm_mu"], data["norm_sd"])
    xhat_test_flat = inverse_features_numpy(fhat_test, data["pca"], data["norm_mu"], data["norm_sd"])
    xhat_train_flat, xhat_test_flat, postprocess_meta = mind.apply_reconstruction_postprocess(
        xhat_train_flat,
        xhat_test_flat,
        data["x_train_flat"],
        cfg,
    )

    x_train_shape = data["x_train_proc"].shape
    x_test_shape = data["x_test_proc"].shape
    xhat_train = xhat_train_flat.reshape(x_train_shape)
    xhat_test = xhat_test_flat.reshape(x_test_shape)

    train_metrics = mind.corr_and_r2(data["x_train_proc"], xhat_train)
    test_metrics = mind.corr_and_r2(data["x_test_proc"], xhat_test)
    event_metrics = mind.compute_event_metrics(
        data["x_test_proc"],
        xhat_test,
        percentile=cfg_float(cfg, "event_metric_percentile", 99.0),
    )

    # Teacher/nonparametric MIND-LLE reference on the same split.
    teacher_train_decoded = teacher["teacher_train_target"]
    teacher_test_decoded = teacher["teacher_test_target"]
    if teacher["decoder_target"] in {"raw", "neural", "activity"}:
        teacher_xhat_train_flat = teacher_train_decoded.astype(np.float32)
        teacher_xhat_test_flat = teacher_test_decoded.astype(np.float32)
    else:
        teacher_xhat_train_flat = inverse_features_numpy(teacher_train_decoded, data["pca"], data["norm_mu"], data["norm_sd"])
        teacher_xhat_test_flat = inverse_features_numpy(teacher_test_decoded, data["pca"], data["norm_mu"], data["norm_sd"])
    teacher_xhat_train_flat, teacher_xhat_test_flat, teacher_post = mind.apply_reconstruction_postprocess(
        teacher_xhat_train_flat,
        teacher_xhat_test_flat,
        data["x_train_flat"],
        cfg,
    )
    teacher_train_metrics = mind.corr_and_r2(data["x_train_proc"], teacher_xhat_train_flat.reshape(x_train_shape))
    teacher_test_metrics = mind.corr_and_r2(data["x_test_proc"], teacher_xhat_test_flat.reshape(x_test_shape))

    print(f"Neural AE train r {train_metrics['r']:.4f} | R2 {train_metrics['R2']:.4f}")
    print(f"Neural AE heldout r {test_metrics['r']:.4f} | R2 {test_metrics['R2']:.4f}")
    print(f"Teacher MIND heldout r {teacher_test_metrics['r']:.4f} | R2 {teacher_test_metrics['R2']:.4f}")
    print(f"Event capture top1 {event_metrics['top_1_percent_event_capture']:.4f} | dyn ratio {event_metrics['pred_dynamics_std_over_true_dynamics_std']:.4f}")

    torch.save({"model_state": model.state_dict(), "config": jsonable(cfg), "train_meta": jsonable(train_meta)}, out_dir / "model.pt")
    np.savez(
        out_dir / "analysis_cache_best.npz",
        x_true=data["x_test_proc"].astype(np.float32),
        x_pred=xhat_test.astype(np.float32),
        x_train_true=data["x_train_proc"].astype(np.float32),
        x_train_pred=xhat_train.astype(np.float32),
        z_test=zhat_test.reshape(x_test_shape[0], x_test_shape[1], -1).astype(np.float32),
        z_train=zhat_train.reshape(x_train_shape[0], x_train_shape[1], -1).astype(np.float32),
        z_teacher_train=teacher["z_train"].reshape(x_train_shape[0], x_train_shape[1], -1).astype(np.float32),
        z_teacher_test=teacher["z_test"].reshape(x_test_shape[0], x_test_shape[1], -1).astype(np.float32),
        train_trial_ids=data["trial_ids"][data["train_idx"]],
        test_trial_ids=data["trial_ids"][data["test_idx"]],
        metrics_json=np.asarray(json.dumps(jsonable({"train": train_metrics, "heldout": test_metrics, "events": event_metrics}))),
    )
    mind.save_recon_heatmap(data["x_test_proc"], xhat_test, out_dir / "raw_vs_recon_t0_15_n0_40.png", "Neural MIND-AE held-out reconstruction")
    mind.save_embedding_plot(teacher["z_lm"], out_dir / "latent_manifold_mds.png")

    final_metrics = {
        "train": train_metrics,
        "heldout": test_metrics,
        "events": event_metrics,
        "teacher_train": teacher_train_metrics,
        "teacher_heldout": teacher_test_metrics,
        "R2": test_metrics["R2"],
        "r": test_metrics["r"],
    }
    torch.save(final_metrics, out_dir / "final_metrics.pt")
    metadata = {
        "script": Path(__file__).name,
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(data["data_path"]),
        "n_trials_total": int(data["sequences"].shape[0]),
        "n_trials_train": int(len(data["train_idx"])),
        "n_trials_heldout": int(len(data["test_idx"])),
        "heldout_fraction": float(len(data["test_idx"]) / data["sequences"].shape[0]),
        "sequence_length": int(data["x_train_proc"].shape[1]),
        "n_neurons": int(data["x_train_proc"].shape[-1]),
        "feature_dim": int(data["f_train_flat"].shape[1]),
        "latent_dim": cfg_int(cfg, "latent_dim", 7),
        "pca_explained_variance": data["pca_var"],
        "globally_silent_neurons_removed": int(data["globally_silent_removed"]),
        "transition_distance_meta": teacher["dist_meta"],
        "embedding_meta": teacher["embed_meta"],
        "landmark_selection": teacher["landmark_meta"],
        "manifold_filter": teacher["manifold_meta"],
        "postprocess_meta": postprocess_meta,
        "teacher_postprocess_meta": teacher_post,
        "training": train_meta,
        "metrics": final_metrics,
        "method_note": "Deterministic neural MIND-AE distilled from train-only v6_mind_lle geometry: PCA features -> neural z -> PCA features, with MIND-LLE z teacher and graph/geodesic local population geometry.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2) + "\n")
    (out_dir / "training_results.txt").write_text(json.dumps(jsonable(final_metrics), indent=2) + "\n")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
