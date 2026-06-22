#!/usr/bin/env python3
"""Raw-frame population-geometry autoencoder.

This v7 variant keeps the research assumption strict:

    raw neural state x_t -> PCA feature f_t -> latent z_t -> reconstruction xhat_t

Position, time, and behavior are never concatenated to the model input. They are
kept only as metadata for weak same-trial position geometry, trial-boundary-aware
transition losses, behavior-preservation probes, and plotting.

The model is designed around the observed E65 data statistics:
- sparse event-like neural activity,
- local same-trial position geometry,
- distributed population geometry rather than clean single-cell place fields,
- rare but strong coactivity pairs,
- strong short-lag temporal continuity.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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


def jsonable(obj: Any) -> Any:
    return neural.jsonable(obj)


def split_raw_frame_trials(
    roi: np.ndarray,
    trials: np.ndarray,
    time: np.ndarray | None,
    position: np.ndarray | None,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return train/test raw frames shaped as [1, frames, neurons].

    Position and time are preserved only as metadata for plotting/diagnostics.
    They are not used as model inputs or geometry features.
    """
    min_frames = cfg_int(cfg, "min_frames", 2)
    trial_source = str(cfg.get("trial_source", "mind_sandbox")).lower()
    if trial_source in {"mind", "mind_sandbox", "sandbox"}:
        requested = np.asarray(cfg.get("mind_sandbox_trials", mind.MIND_SANDBOX_TRIALS), dtype=np.int64)
        trial_ids = []
        for tid in requested:
            idx = np.flatnonzero(trials == tid)
            if idx.size >= min_frames:
                trial_ids.append(int(tid))
        trial_ids = np.asarray(trial_ids, dtype=np.int64)
    else:
        trial_ids, groups = mind.group_indices_by_trial(
            trials,
            drop_first_trials=cfg_int(cfg, "drop_first_trials", 0),
            min_frames=min_frames,
        )
        trial_ids = trial_ids.astype(np.int64)
    if trial_ids.size == 0:
        raise ValueError("No usable raw-frame trials found.")

    frac = cfg_float(cfg, "mind_test_frac", cfg_float(cfg, "test_frac", 0.1))
    seed = cfg_int(cfg, "mind_split_seed", cfg_int(cfg, "seed", 42))
    rng = np.random.default_rng(seed)
    train_mask = rng.random(trial_ids.size) > frac
    if train_mask.all():
        train_mask[rng.integers(0, trial_ids.size)] = False
    if (~train_mask).all():
        train_mask[rng.integers(0, trial_ids.size)] = True
    train_trials = trial_ids[train_mask]
    test_trials = trial_ids[~train_mask]

    frame_mask = np.isin(trials.astype(np.int64), trial_ids)
    if cfg_bool(cfg, "filter_silent_frames", True):
        frame_score = np.sum(roi, axis=1)
        frame_mask &= frame_score > cfg_float(cfg, "silent_frame_threshold", 0.0)

    all_data = roi[frame_mask].astype(np.float32)
    all_trial_ids = trials[frame_mask].astype(np.int64)
    all_frame_index = np.flatnonzero(frame_mask).astype(np.int64)
    if time is None:
        all_time = all_frame_index.astype(np.float32)
    else:
        all_time = np.asarray(time[frame_mask], dtype=np.float32)
    if position is None:
        all_position = np.full(all_data.shape[0], np.nan, dtype=np.float32)
    else:
        all_position = np.asarray(position[frame_mask], dtype=np.float32)

    train_frame_mask = np.isin(all_trial_ids, train_trials)
    test_frame_mask = np.isin(all_trial_ids, test_trials)
    x_train = all_data[train_frame_mask]
    x_test = all_data[test_frame_mask]
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("Raw-frame split produced empty train or heldout data.")

    meta = {
        "data_mode": "raw_frames_by_trial",
        "trial_source": trial_source,
        "trial_ids": trial_ids,
        "train_trial_ids": train_trials,
        "test_trial_ids": test_trials,
        "train_frame_trial_ids": all_trial_ids[train_frame_mask],
        "test_frame_trial_ids": all_trial_ids[test_frame_mask],
        "train_frame_time": all_time[train_frame_mask],
        "test_frame_time": all_time[test_frame_mask],
        "train_frame_position": all_position[train_frame_mask],
        "test_frame_position": all_position[test_frame_mask],
        "train_source_frame_index": all_frame_index[train_frame_mask],
        "test_source_frame_index": all_frame_index[test_frame_mask],
        "n_trials_total": int(trial_ids.size),
        "n_trials_train": int(train_trials.size),
        "n_trials_heldout": int(test_trials.size),
        "heldout_fraction": float(test_trials.size / trial_ids.size),
        "n_train_frames": int(x_train.shape[0]),
        "n_test_frames": int(x_test.shape[0]),
        "filter_silent_frames": cfg_bool(cfg, "filter_silent_frames", True),
        "note": "Raw neural frames; position/time retained as metadata only.",
    }
    return x_train[None, :, :], x_test[None, :, :], meta


def load_frame_covariates(data_path: Path, frame_indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Load frame-aligned behavioral/task arrays for diagnostics only."""
    requested = {
        "position": ["Position", "position"],
        "velocity": ["Velocity", "velocity"],
        "choice": ["Choice", "choice"],
        "correct": ["ChoiceCorrect", "Correct", "choice_correct"],
        "maze_id": ["MazeID", "Maze", "maze_id"],
        "trial_type": ["TrialType", "trial_type"],
        "evidence": ["Evidence", "evidence"],
        "time": ["Time", "time"],
    }
    covariates: Dict[str, np.ndarray] = {}
    with np.load(data_path, allow_pickle=True) as npz:
        for out_name, names in requested.items():
            arr = mind.get_npz_array(npz, names)
            if arr is None:
                continue
            arr = np.asarray(arr).reshape(-1)
            if frame_indices.size == 0 or arr.size <= int(np.max(frame_indices)):
                continue
            covariates[out_name] = arr[frame_indices].astype(np.float32)
    return covariates


def boundary_deltas(values: np.ndarray, frame_trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute next-frame deltas only within the same trial."""
    values = np.asarray(values, dtype=np.float32)
    ids = np.asarray(frame_trial_ids).reshape(-1)
    if values.shape[0] != ids.size:
        raise ValueError(f"values/frame_trial_ids mismatch: {values.shape[0]} vs {ids.size}")
    deltas = np.zeros_like(values, dtype=np.float32)
    valid = np.zeros(values.shape[0], dtype=bool)
    if values.shape[0] > 1:
        same_trial = ids[:-1] == ids[1:]
        deltas[:-1][same_trial] = values[1:][same_trial] - values[:-1][same_trial]
        valid[:-1] = same_trial
    return deltas, valid


class PopulationGeometryAE(nn.Module):
    """Feature autoencoder with sparse event raw decoder and next-state head."""

    def __init__(
        self,
        feature_dim: int,
        raw_dim: int,
        latent_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        transition_layers: int,
    ):
        super().__init__()
        self.encoder = neural.MindDistilledAE._mlp(feature_dim, latent_dim, hidden, layers, dropout)
        self.decoder = neural.MindDistilledAE._mlp(latent_dim, feature_dim, hidden, layers, dropout)
        self.event_logits = neural.MindDistilledAE._mlp(latent_dim, raw_dim, hidden, layers, dropout)
        self.event_amplitude = neural.MindDistilledAE._mlp(latent_dim, raw_dim, hidden, layers, dropout)
        self.transition_head = neural.MindDistilledAE._mlp(latent_dim, feature_dim, hidden, transition_layers, dropout)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(features)
        fhat = self.decoder(z)
        logits = self.event_logits(z)
        amplitude = F.softplus(self.event_amplitude(z))
        xhat = torch.sigmoid(logits) * amplitude
        fnext_hat = self.transition_head(z)
        return fhat, z, fnext_hat, xhat, logits, amplitude


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

    x_train, x_test, raw_meta = split_raw_frame_trials(roi, trials, time, position, cfg)
    print(
        "Raw-frame split: "
        f"train trials={raw_meta['n_trials_train']} ({raw_meta['n_train_frames']} frames) | "
        f"heldout trials={raw_meta['n_trials_heldout']} ({raw_meta['n_test_frames']} frames)"
    )
    train_covariates = load_frame_covariates(data_path, raw_meta["train_source_frame_index"])
    test_covariates = load_frame_covariates(data_path, raw_meta["test_source_frame_index"])

    if cfg_bool(cfg, "filter_train_silent_neurons", False):
        std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        keep = std > cfg_float(cfg, "train_silent_threshold", 1e-7)
        removed_train = int((~keep).sum())
        x_train = x_train[..., keep]
        x_test = x_test[..., keep]
        mask_indices = np.flatnonzero(mask)
        mask[mask_indices[~keep]] = False
        print(f"Train-std neuron filter: removed {removed_train}; kept {x_train.shape[-1]} neurons.")

    x_train_proc = x_train.astype(np.float32)
    x_test_proc = x_test.astype(np.float32)
    x_train_flat = x_train_proc.reshape(-1, x_train_proc.shape[-1])
    x_test_flat = x_test_proc.reshape(-1, x_test_proc.shape[-1])

    norm_mu = None
    norm_sd = None
    if cfg_bool(cfg, "normalize_inputs", False):
        norm_mu = x_train_flat.mean(axis=0, keepdims=True).astype(np.float32)
        norm_sd = x_train_flat.std(axis=0, keepdims=True).astype(np.float32)
        norm_sd = np.where(norm_sd < cfg_float(cfg, "normalize_eps", 1e-6), 1.0, norm_sd)
        x_train_for_pca = ((x_train_flat - norm_mu) / norm_sd).astype(np.float32)
        x_test_for_pca = ((x_test_flat - norm_mu) / norm_sd).astype(np.float32)
    else:
        x_train_for_pca = x_train_flat.astype(np.float32)
        x_test_for_pca = x_test_flat.astype(np.float32)

    use_pca = cfg_bool(cfg, "use_pca", True) or cfg_int(cfg, "pca_dim", 0) > 0 or cfg.get("pca_variance", None) is not None
    pca = None
    if use_pca:
        pca_dim = cfg.get("pca_dim", 0)
        pca_variance = cfg.get("pca_variance", None)
        if isinstance(pca_dim, (int, float)) and float(pca_dim) > 0:
            n_components: int | float = min(int(pca_dim), x_train_for_pca.shape[1])
        elif pca_variance is not None and 0.0 < float(pca_variance) < 1.0:
            n_components = float(pca_variance)
        else:
            n_components = min(x_train_for_pca.shape[1], cfg_int(cfg, "pca_max_dim", 64))
        pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
        f_train_flat = pca.fit_transform(x_train_for_pca).astype(np.float32)
        f_test_flat = pca.transform(x_test_for_pca).astype(np.float32)
        pca_var = float(np.sum(pca.explained_variance_ratio_))
        print(f"PCA feature space: dim={f_train_flat.shape[1]}, explained_variance={pca_var:.4f}")
    else:
        f_train_flat = x_train_for_pca
        f_test_flat = x_test_for_pca
        pca_var = float("nan")
        print(f"Feature space: raw neurons, dim={f_train_flat.shape[1]}")

    return {
        "rng": rng,
        "data_path": data_path,
        "mask": mask,
        "globally_silent_removed": removed,
        "data_mode": "raw_frames_by_trial",
        "raw_split_meta": raw_meta,
        "train_covariates": train_covariates,
        "test_covariates": test_covariates,
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


def build_teacher(cfg: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    rng = data["rng"]
    raw_meta = data["raw_split_meta"]
    f_train = data["f_train_flat"]
    x_train = data["x_train_flat"]

    if cfg_bool(cfg, "respect_trial_boundaries", True):
        delta_f, valid_transition = boundary_deltas(f_train, raw_meta["train_frame_trial_ids"])
        delta_x, _ = boundary_deltas(x_train, raw_meta["train_frame_trial_ids"])
    else:
        delta_f = mind.sequence_deltas(f_train[None, :, :]).reshape(f_train.shape)
        delta_x = mind.sequence_deltas(x_train[None, :, :]).reshape(x_train.shape)
        valid_transition = np.ones(f_train.shape[0], dtype=bool)

    manifold_mask = np.ones(f_train.shape[0], dtype=bool)
    manifold_meta: Dict[str, Any] = {
        "filter_silent_frames_for_manifold": cfg_bool(cfg, "filter_silent_frames_for_manifold", False),
        "respect_trial_boundaries": cfg_bool(cfg, "respect_trial_boundaries", True),
        "valid_transition_states": int(valid_transition.sum()),
        "total_train_states": int(f_train.shape[0]),
    }
    if cfg_bool(cfg, "filter_silent_frames_for_manifold", False):
        score_mode = str(cfg.get("silent_frame_score", "sum"))
        threshold = cfg_float(cfg, "silent_frame_threshold", 0.0)
        frame_score = mind.score_activity(x_train, score_mode)
        manifold_mask = frame_score > threshold
        manifold_meta.update({
            "silent_frame_score": score_mode,
            "silent_frame_threshold": threshold,
            "active_train_states": int(manifold_mask.sum()),
            "removed_silent_train_states": int((~manifold_mask).sum()),
            "active_fraction": float(manifold_mask.mean()),
        })

    if cfg_bool(cfg, "exclude_boundary_landmarks", True):
        landmark_pool_mask = manifold_mask & valid_transition
    else:
        landmark_pool_mask = manifold_mask
    if not np.any(landmark_pool_mask):
        landmark_pool_mask = manifold_mask
    pool_indices = np.flatnonzero(landmark_pool_mask)
    landmark_count = cfg_int(cfg, "landmark_count", min(750, pool_indices.size))
    lm_local, landmark_meta = mind.select_landmarks(
        f_train[landmark_pool_mask],
        x_train[landmark_pool_mask],
        delta_x[landmark_pool_mask],
        landmark_count,
        rng,
        cfg,
    )
    lm_idx = pool_indices[lm_local]
    f_lm = f_train[lm_idx]
    df_lm = delta_f[lm_idx]
    print(
        f"Selected train landmarks: {f_lm.shape[0]} of {f_train.shape[0]} raw states "
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
    print(f"Built neural-state distances: max={dist_meta['distance_max']:.4f}, disconnected_replaced={dist_meta['disconnected_pairs_replaced']}")

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

    inverse_decoder = str(cfg.get("inverse_decoder", "kernel")).lower()
    inverse_k = cfg_int(cfg, "inverse_k", cfg_int(cfg, "kernel_k", lle_k))
    kernel_bandwidth = cfg_float(cfg, "kernel_bandwidth", 0.35)
    decoder_target = str(cfg.get("decoder_target", "feature")).lower()
    decoder_train_target = data["x_train_flat"] if decoder_target in {"raw", "neural", "activity"} else data["f_train_flat"]
    if inverse_decoder in {"kernel", "rbf", "weighted", "weighted_average"}:
        z2target = mind.KernelRegressor(inverse_k, kernel_bandwidth).fit(z_train, decoder_train_target)
    elif inverse_decoder in {"lle", "local_linear"}:
        z2target = mind.LLEMapper(inverse_k, lle_ridge).fit(z_train, decoder_train_target)
    else:
        raise ValueError(f"Unknown inverse_decoder={inverse_decoder!r}")

    return {
        "lm_idx": lm_idx,
        "z_lm": z_lm,
        "d_mind": d_mind,
        "z_train": z_train.astype(np.float32),
        "z_test": z_test.astype(np.float32),
        "z_train_norm": z_train_norm,
        "z_mu": z_mu,
        "z_sd": z_sd,
        "teacher_train_target": z2target.transform(z_train).astype(np.float32),
        "teacher_test_target": z2target.transform(z_test).astype(np.float32),
        "decoder_target": decoder_target,
        "dist_meta": dist_meta,
        "embed_meta": embed_meta,
        "landmark_meta": landmark_meta,
        "manifold_meta": manifold_meta,
    }


def make_event_weights(x: np.ndarray, frame_trial_ids: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    alpha = cfg_float(cfg, "event_weight_alpha", 0.0)
    if alpha <= 0:
        return np.ones((x.shape[0], 1), dtype=np.float32)
    mode = str(cfg.get("event_weight_mode", "activity_or_dx")).lower()
    activity = np.max(np.maximum(x, 0.0), axis=1)
    dx, _ = boundary_deltas(x, frame_trial_ids)
    dyn = np.max(np.abs(dx), axis=1)
    if mode == "activity":
        score = activity
    elif mode in {"dx", "dynamics", "transition"}:
        score = dyn
    else:
        score = np.maximum(activity, dyn)
    scale = np.percentile(score, cfg_float(cfg, "event_weight_percentile", 95.0))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.max(score) if np.max(score) > 1e-8 else 1.0
    score = np.clip(score / scale, 0.0, 10.0)
    return (1.0 + alpha * score[:, None]).astype(np.float32)


def event_sampling_weights(x: np.ndarray, frame_trial_ids: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Oversample rare event and transition frames without changing inputs."""
    if not cfg_bool(cfg, "event_balanced_sampling", True):
        return np.ones(x.shape[0], dtype=np.float32)
    activity = np.max(np.maximum(x, 0.0), axis=1)
    dx, valid = boundary_deltas(x, frame_trial_ids)
    dyn = np.max(np.abs(dx), axis=1)
    high_activity = activity >= np.percentile(activity, cfg_float(cfg, "sampling_activity_percentile", 95.0))
    high_dx = dyn >= np.percentile(dyn[valid] if np.any(valid) else dyn, cfg_float(cfg, "sampling_dx_percentile", 95.0))
    low_activity = activity <= np.percentile(activity, cfg_float(cfg, "sampling_low_percentile", 50.0))
    weights = np.ones(x.shape[0], dtype=np.float32)
    weights[high_activity] *= cfg_float(cfg, "sampling_high_activity_boost", 3.0)
    weights[high_dx] *= cfg_float(cfg, "sampling_high_dx_boost", 3.0)
    weights[low_activity] *= cfg_float(cfg, "sampling_low_activity_boost", 1.25)
    weights = np.maximum(weights, 1e-3)
    return weights.astype(np.float32)


def build_position_triplets(
    positions: np.ndarray,
    frame_trial_ids: np.ndarray,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = positions.size
    pos_idx = np.arange(n, dtype=np.int64)
    neg_idx = np.arange(n, dtype=np.int64)
    if n == 0 or not np.all(np.isfinite(positions)):
        return pos_idx, neg_idx, {"enabled": False, "reason": "missing_or_nonfinite_position"}

    r_pos = cfg_float(cfg, "position_positive_radius_cm", 10.0)
    r_neg = cfg_float(cfg, "position_negative_radius_cm", 50.0)
    fallback = 0
    for tid in np.unique(frame_trial_ids):
        idx = np.flatnonzero(frame_trial_ids == tid)
        p = positions[idx].astype(np.float64)
        for local_i, global_i in enumerate(idx):
            dist = np.abs(p - p[local_i])
            pos_pool = idx[(dist > 0) & (dist <= r_pos)]
            neg_pool = idx[dist >= r_neg]
            if pos_pool.size == 0:
                order = np.argsort(dist)
                pos_pool = idx[order[1:2]] if order.size > 1 else idx[order[:1]]
                fallback += 1
            if neg_pool.size == 0:
                order = np.argsort(dist)[::-1]
                neg_pool = idx[order[:1]]
                fallback += 1
            pos_idx[global_i] = int(rng.choice(pos_pool))
            neg_idx[global_i] = int(rng.choice(neg_pool))
    return pos_idx, neg_idx, {
        "enabled": True,
        "positive_radius_cm": float(r_pos),
        "negative_radius_cm": float(r_neg),
        "fallback_assignments": int(fallback),
        "n_triplets": int(n),
    }


def _lle_weights(xi: np.ndarray, nb: np.ndarray, ridge: float) -> np.ndarray:
    centered = nb - xi[None, :]
    gram = centered @ centered.T
    trace = float(np.trace(gram))
    reg = ridge * (trace / max(gram.shape[0], 1) if trace > 0 else 1.0)
    gram = gram + np.eye(gram.shape[0], dtype=np.float32) * (reg + 1e-8)
    ones = np.ones(gram.shape[0], dtype=np.float32)
    try:
        w = np.linalg.solve(gram, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(gram, ones, rcond=None)[0]
    s = float(np.sum(w))
    if abs(s) < 1e-12:
        return np.full(w.shape, 1.0 / max(w.size, 1), dtype=np.float32)
    return (w / s).astype(np.float32)


def build_local_lle_targets(features: np.ndarray, cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    k = cfg_int(cfg, "local_lle_k", cfg_int(cfg, "lle_k", 16))
    ridge = cfg_float(cfg, "local_lle_ridge", cfg_float(cfg, "lle_ridge", 0.001))
    k = min(max(1, k), max(1, features.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(features)
    inds = nn.kneighbors(features, return_distance=False)[:, 1:]
    weights = np.empty((features.shape[0], k), dtype=np.float32)
    for i, row in enumerate(inds):
        weights[i] = _lle_weights(features[i], features[row], ridge)
    return inds.astype(np.int64), weights, {"enabled": True, "k": int(k), "ridge": float(ridge)}


def select_coactivity_pairs(x: np.ndarray, cfg: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    count = cfg_int(cfg, "coactivity_pair_count", 32)
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64), {"enabled": False, "pair_count": 0}
    positive_frac = np.mean(x > cfg_float(cfg, "sparse_event_threshold", 0.0), axis=0)
    keep = positive_frac >= cfg_float(cfg, "coactivity_min_active_fraction", 0.001)
    if keep.sum() < 2:
        return np.zeros((0, 2), dtype=np.int64), {"enabled": False, "reason": "too_few_active_neurons"}
    xk = x[:, keep].astype(np.float64)
    corr = np.corrcoef(xk, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(corr.shape[0], k=1)
    vals = np.abs(corr[iu])
    order = np.argsort(vals)[::-1][:count]
    kept_neurons = np.flatnonzero(keep)
    pairs = np.column_stack([kept_neurons[iu[0][order]], kept_neurons[iu[1][order]]]).astype(np.int64)
    return pairs, {
        "enabled": True,
        "pair_count": int(pairs.shape[0]),
        "min_active_fraction": float(cfg_float(cfg, "coactivity_min_active_fraction", 0.001)),
        "max_abs_corr": float(vals[order[0]]) if order.size else 0.0,
        "mean_abs_corr_selected": float(np.mean(vals[order])) if order.size else 0.0,
    }


def corr_matrix_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    x = x / torch.clamp(x.std(dim=0, keepdim=True), min=eps)
    return (x.T @ x) / max(int(x.shape[0]) - 1, 1)


def population_corr_loss(xhat: torch.Tensor, x: torch.Tensor, neuron_idx: torch.Tensor) -> torch.Tensor:
    if neuron_idx.numel() < 2 or x.shape[0] < 3:
        return torch.zeros((), device=x.device)
    true_corr = corr_matrix_torch(x[:, neuron_idx])
    pred_corr = corr_matrix_torch(xhat[:, neuron_idx])
    return F.mse_loss(pred_corr, true_corr)


def coactivity_loss(xhat: torch.Tensor, x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.zeros((), device=x.device)
    true = x[:, pairs[:, 0]] * x[:, pairs[:, 1]]
    pred = xhat[:, pairs[:, 0]] * xhat[:, pairs[:, 1]]
    return F.mse_loss(pred, true)


def stratified_reconstruction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frame_trial_ids: np.ndarray,
    percentile: float = 95.0,
) -> Dict[str, Any]:
    true = y_true.reshape(-1, y_true.shape[-1])
    pred = y_pred.reshape(-1, y_pred.shape[-1])
    activity = np.max(np.maximum(true, 0.0), axis=1)
    dx, valid = boundary_deltas(true, frame_trial_ids)
    dyn = np.max(np.abs(dx), axis=1)
    out: Dict[str, Any] = {}
    masks = {
        "low_activity": activity <= np.percentile(activity, 50.0),
        "top_5pct_activity": activity >= np.percentile(activity, 95.0),
        "top_1pct_activity": activity >= np.percentile(activity, 99.0),
        "high_dx": dyn >= np.percentile(dyn[valid] if np.any(valid) else dyn, percentile),
    }
    for name, mask in masks.items():
        if np.any(mask):
            out[name] = mind.corr_and_r2(true[mask], pred[mask])
            out[name]["n_frames"] = int(mask.sum())
        else:
            out[name] = {"n_frames": 0}
    return out


def behavior_preservation_metrics(
    f_train: np.ndarray,
    f_test: np.ndarray,
    fhat_test: np.ndarray,
    train_covariates: Dict[str, np.ndarray],
    test_covariates: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    continuous = {"position", "velocity", "evidence", "time"}
    categorical = {"choice", "correct", "maze_id", "trial_type"}
    out: Dict[str, Any] = {}
    for name, y_train in train_covariates.items():
        if name not in test_covariates:
            continue
        y_test = np.asarray(test_covariates[name]).reshape(-1)
        y_train = np.asarray(y_train).reshape(-1)
        finite_train = np.isfinite(y_train)
        finite_test = np.isfinite(y_test)
        if finite_train.sum() < 10 or finite_test.sum() < 10:
            continue
        if name in continuous:
            model = Ridge(alpha=1.0)
            model.fit(f_train[finite_train], y_train[finite_train])
            pred_true = model.predict(f_test[finite_test])
            pred_recon = model.predict(fhat_test[finite_test])
            out[name] = {
                "type": "continuous_probe",
                "true_feature_R2": float(r2_score(y_test[finite_test], pred_true)),
                "recon_feature_R2": float(r2_score(y_test[finite_test], pred_recon)),
            }
        elif name in categorical:
            classes = np.unique(y_train[finite_train])
            if classes.size < 2 or classes.size > 20:
                continue
            model = LogisticRegression(max_iter=500, class_weight="balanced")
            model.fit(f_train[finite_train], y_train[finite_train].astype(int))
            pred_true = model.predict(f_test[finite_test])
            pred_recon = model.predict(fhat_test[finite_test])
            out[name] = {
                "type": "categorical_probe",
                "true_feature_balanced_accuracy": float(balanced_accuracy_score(y_test[finite_test].astype(int), pred_true)),
                "recon_feature_balanced_accuracy": float(balanced_accuracy_score(y_test[finite_test].astype(int), pred_recon)),
            }
    return out


def save_raw_frame_recon_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    frame_trial_ids: np.ndarray,
    frame_time: np.ndarray,
    trial_index: int = 0,
    frame_start: int = 0,
    frame_end: int = 15,
    neuron_start: int = 0,
    neuron_end: int = 40,
) -> None:
    """Plot one heldout raw-frame trial against elapsed time in seconds."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ids = np.asarray(frame_trial_ids).reshape(-1)
    times = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != 1 or y_pred.shape != y_true.shape:
        raise ValueError(f"Expected matching [1, frames, neurons] arrays, got {y_true.shape} and {y_pred.shape}")
    if ids.size != y_true.shape[1] or times.size != y_true.shape[1]:
        raise ValueError("frame_trial_ids/frame_time must have one entry per frame")

    unique_ids = np.unique(ids)
    pick = int(np.clip(trial_index, 0, unique_ids.size - 1))
    trial_id = unique_ids[pick]
    frame_idx = np.flatnonzero(ids == trial_id)
    true_trial = y_true[0, frame_idx, :]
    pred_trial = y_pred[0, frame_idx, :]
    time_trial = times[frame_idx] - times[frame_idx][0]

    f0 = max(0, int(frame_start))
    f1 = min(int(frame_end) + 1, true_trial.shape[0])
    n0 = max(0, int(neuron_start))
    n1 = min(int(neuron_end) + 1, true_trial.shape[1])
    if f0 >= f1 or n0 >= n1:
        raise ValueError(
            f"Requested empty raw-frame window: frames {frame_start}-{frame_end}, "
            f"neurons {neuron_start}-{neuron_end}, trial shape={true_trial.shape}"
        )

    raw = true_trial[f0:f1, n0:n1].T
    recon = pred_trial[f0:f1, n0:n1].T
    resid = raw - recon
    x = time_trial[f0:f1]
    if x.size > 1:
        x_edges = np.empty(x.size + 1, dtype=np.float64)
        x_edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        x_edges[0] = x[0] - 0.5 * (x[1] - x[0])
        x_edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    else:
        x_edges = np.asarray([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
    x_edges = np.maximum.accumulate(x_edges)
    y_edges = np.arange(n0, n1 + 1) - 0.5

    activity_vmin = float(np.nanmin([np.nanmin(raw), np.nanmin(recon)]))
    activity_vmax = float(np.nanmax([np.nanmax(raw), np.nanmax(recon)]))
    if not np.isfinite(activity_vmin) or not np.isfinite(activity_vmax) or activity_vmin == activity_vmax:
        activity_vmin, activity_vmax = 0.0, 1.0
    resid_vmax = float(np.nanmax(np.abs(resid))) if resid.size else 1.0
    if not np.isfinite(resid_vmax) or resid_vmax <= 0:
        resid_vmax = 1.0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, arr, label in zip(axes, [raw, recon, resid], ["raw", "recon", "raw - recon"]):
        if label == "raw - recon":
            im = ax.pcolormesh(x_edges, y_edges, arr, cmap="coolwarm", vmin=-resid_vmax, vmax=resid_vmax, shading="flat")
        else:
            im = ax.pcolormesh(x_edges, y_edges, arr, cmap="viridis", vmin=activity_vmin, vmax=activity_vmax, shading="flat")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("neuron")
        ax.set_ylim(n1 - 1, n0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title}: heldout trial {int(trial_id)}, frames {f0}-{f1 - 1}, neurons {n0}-{n1 - 1}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_model(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    teacher: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    f_train = data["f_train_flat"].astype(np.float32)
    z_teacher = teacher["z_train_norm"].astype(np.float32)
    teacher_target = teacher["teacher_train_target"].astype(np.float32)
    if teacher_target.shape[1] != f_train.shape[1]:
        print("Teacher target is not feature-space; disabling teacher reconstruction loss.")
        teacher_target = f_train
        cfg["lambda_teacher_recon"] = 0.0
    x_train = data["x_train_flat"].astype(np.float32)
    weights = make_event_weights(x_train, data["raw_split_meta"]["train_frame_trial_ids"], cfg)
    delta_f_train, valid_next_train = boundary_deltas(f_train, data["raw_split_meta"]["train_frame_trial_ids"])
    delta_x_train, _ = boundary_deltas(x_train, data["raw_split_meta"]["train_frame_trial_ids"])
    f_next_train = (f_train + delta_f_train).astype(np.float32)
    x_next_train = (x_train + delta_x_train).astype(np.float32)
    next_mask_train = valid_next_train.astype(np.float32)[:, None]
    rng = np.random.default_rng(cfg_int(cfg, "seed", 42) + 17)

    pos_idx, neg_idx, position_meta = build_position_triplets(
        data["raw_split_meta"]["train_frame_position"].astype(np.float32),
        data["raw_split_meta"]["train_frame_trial_ids"],
        cfg,
        rng,
    )
    local_idx, local_w, local_meta = build_local_lle_targets(f_train, cfg)
    coactivity_pairs, coactivity_meta = select_coactivity_pairs(x_train, cfg)
    pop_count = min(cfg_int(cfg, "pop_corr_neuron_count", 128), x_train.shape[1])
    pop_neuron_idx = np.argsort(x_train.var(axis=0))[::-1][:pop_count].astype(np.int64)
    event_rate = np.mean(x_train > cfg_float(cfg, "sparse_event_threshold", 0.0), axis=0, keepdims=True).astype(np.float32)
    pos_weight_value = (1.0 - event_rate) / np.maximum(event_rate, 1e-4)
    pos_weight_value = np.clip(pos_weight_value, 1.0, cfg_float(cfg, "event_bce_pos_weight_max", 50.0)).astype(np.float32)
    sample_weights_all = event_sampling_weights(x_train, data["raw_split_meta"]["train_frame_trial_ids"], cfg)

    n = f_train.shape[0]
    order = rng.permutation(n)
    n_val = int(round(n * cfg_float(cfg, "state_val_frac", 0.1)))
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n_val > 0 else order

    tensors = {
        "f": torch.from_numpy(f_train),
        "z": torch.from_numpy(z_teacher),
        "teacher": torch.from_numpy(teacher_target),
        "x": torch.from_numpy(x_train),
        "w": torch.from_numpy(weights),
        "f_next": torch.from_numpy(f_next_train),
        "x_next": torch.from_numpy(x_next_train),
        "next_mask": torch.from_numpy(next_mask_train),
        "f_pos": torch.from_numpy(f_train[pos_idx]),
        "f_neg": torch.from_numpy(f_train[neg_idx]),
        "f_local": torch.from_numpy(f_train[local_idx]),
        "w_local": torch.from_numpy(local_w),
    }

    def subset(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(*(t[indices] for t in tensors.values()))

    batch_size = cfg_int(cfg, "batch_size", 512)
    if cfg_bool(cfg, "event_balanced_sampling", True):
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights_all[train_idx].astype(np.float64)),
            num_samples=int(train_idx.size),
            replacement=True,
        )
        train_loader = DataLoader(subset(train_idx), batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(subset(train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(subset(val_idx), batch_size=batch_size, shuffle=False, drop_last=False) if n_val > 0 else None

    model: nn.Module = PopulationGeometryAE(
        feature_dim=f_train.shape[1],
        raw_dim=x_train.shape[1],
        latent_dim=cfg_int(cfg, "latent_dim", 7),
        hidden=cfg_int(cfg, "hidden", 256),
        layers=cfg_int(cfg, "layers", 2),
        dropout=cfg_float(cfg, "dropout", 0.0),
        transition_layers=cfg_int(cfg, "transition_head_layers", cfg_int(cfg, "layers", 2)),
    ).to(device)

    pca = data["pca"]
    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device) if pca is not None else None
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device) if pca is not None else None
    norm_mu = torch.tensor(data["norm_mu"], dtype=torch.float32, device=device) if data["norm_mu"] is not None else None
    norm_sd = torch.tensor(data["norm_sd"], dtype=torch.float32, device=device) if data["norm_sd"] is not None else None
    event_rate_t = torch.tensor(event_rate, dtype=torch.float32, device=device)
    pos_weight_t = torch.tensor(pos_weight_value.reshape(-1), dtype=torch.float32, device=device)
    pop_idx_t = torch.tensor(pop_neuron_idx, dtype=torch.long, device=device)
    coactivity_pairs_t = torch.tensor(coactivity_pairs, dtype=torch.long, device=device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_float(cfg, "lr", 1e-3),
        weight_decay=cfg_float(cfg, "weight_decay", 1e-5),
    )
    lambda_feature = cfg_float(cfg, "lambda_feature_recon", 1.0)
    lambda_z = cfg_float(cfg, "lambda_teacher_z", 0.0)
    lambda_teacher = cfg_float(cfg, "lambda_teacher_recon", 0.25)
    lambda_raw = cfg_float(cfg, "lambda_raw_recon", 0.1)
    lambda_next = cfg_float(cfg, "lambda_next", 0.0)
    lambda_next_raw = cfg_float(cfg, "lambda_next_raw", 0.0)
    lambda_dx = cfg_float(cfg, "lambda_dx", 0.0)
    lambda_pos = cfg_float(cfg, "lambda_position_geometry", 0.0)
    lambda_local = cfg_float(cfg, "lambda_local_geometry", 0.0)
    lambda_rate = cfg_float(cfg, "lambda_event_rate", 0.0)
    lambda_coact = cfg_float(cfg, "lambda_coactivity", 0.0)
    lambda_popcorr = cfg_float(cfg, "lambda_population_corr", 0.0)
    lambda_event_bce = cfg_float(cfg, "lambda_event_bce", 0.05)
    lambda_event_amp = cfg_float(cfg, "lambda_event_amplitude", 0.1)
    event_threshold = cfg_float(cfg, "sparse_event_threshold", 0.0)

    def loss_on_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, Dict[str, float]]:
        (
            f,
            zt,
            ft,
            x,
            w,
            f_next,
            x_next,
            next_mask,
            f_pos,
            f_neg,
            f_local,
            w_local,
        ) = (b.to(device) for b in batch)
        fhat, zhat, fnext_hat, xhat, logits, amplitude = model(f)
        loss_feature = F.mse_loss(fhat, f)
        loss_z = F.mse_loss(zhat, zt)
        loss_teacher = F.mse_loss(fhat, ft)
        target_event = (x > event_threshold).to(x.dtype)
        loss_event_bce = F.binary_cross_entropy_with_logits(logits, target_event, pos_weight=pos_weight_t)
        positive = target_event > 0
        if bool(torch.any(positive)):
            amp_err = (amplitude - x) ** 2
            loss_amp = torch.mean(w.expand_as(x)[positive] * amp_err[positive])
        else:
            loss_amp = torch.zeros((), device=device)
        loss_raw_mse = torch.mean(w * (xhat - x) ** 2)
        loss_raw = loss_raw_mse + lambda_event_bce * loss_event_bce + lambda_event_amp * loss_amp
        loss_rate = F.mse_loss(torch.sigmoid(logits).mean(dim=0, keepdim=True), event_rate_t)
        if lambda_next > 0:
            next_denom = torch.clamp(next_mask.sum() * f_next.shape[1], min=1.0)
            loss_next = (((fnext_hat - f_next) ** 2) * next_mask).sum() / next_denom
        else:
            loss_next = torch.zeros((), device=device)
        if lambda_next_raw > 0 or lambda_dx > 0:
            xnext_hat = neural.inverse_features_torch(fnext_hat, pca_components, pca_mean, norm_mu, norm_sd)
            xnext_hat = torch.clamp(xnext_hat, min=0.0)
        else:
            xnext_hat = None
        if xnext_hat is not None and lambda_next_raw > 0:
            next_raw_denom = torch.clamp(next_mask.sum() * x_next.shape[1], min=1.0)
            loss_next_raw = (((xnext_hat - x_next) ** 2) * next_mask).sum() / next_raw_denom
        else:
            loss_next_raw = torch.zeros((), device=device)
        if xnext_hat is not None and lambda_dx > 0:
            dx_true = x_next - x
            dx_pred = xnext_hat - xhat
            dx_denom = torch.clamp(next_mask.sum() * x.shape[1], min=1.0)
            loss_dx = (((dx_pred - dx_true) ** 2) * next_mask).sum() / dx_denom
        else:
            loss_dx = torch.zeros((), device=device)
        if lambda_pos > 0:
            z_pos = model.encoder(f_pos)
            z_neg = model.encoder(f_neg)
            d_pos = torch.sum((zhat - z_pos) ** 2, dim=1)
            d_neg = torch.sum((zhat - z_neg) ** 2, dim=1)
            loss_pos = torch.mean(F.relu(d_pos - d_neg + cfg_float(cfg, "position_geometry_margin", 0.2)))
        else:
            loss_pos = torch.zeros((), device=device)
        if lambda_local > 0:
            bsz, k, fdim = f_local.shape
            z_local = model.encoder(f_local.reshape(bsz * k, fdim)).reshape(bsz, k, -1)
            z_recon = torch.sum(w_local[:, :, None] * z_local, dim=1)
            loss_local = F.mse_loss(zhat, z_recon)
        else:
            loss_local = torch.zeros((), device=device)
        loss_popcorr = population_corr_loss(xhat, x, pop_idx_t) if lambda_popcorr > 0 else torch.zeros((), device=device)
        loss_coact = coactivity_loss(xhat, x, coactivity_pairs_t) if lambda_coact > 0 else torch.zeros((), device=device)
        loss = (
            lambda_feature * loss_feature
            + lambda_z * loss_z
            + lambda_teacher * loss_teacher
            + lambda_raw * loss_raw
            + lambda_next * loss_next
            + lambda_next_raw * loss_next_raw
            + lambda_dx * loss_dx
            + lambda_pos * loss_pos
            + lambda_local * loss_local
            + lambda_rate * loss_rate
            + lambda_coact * loss_coact
            + lambda_popcorr * loss_popcorr
        )
        return loss, {
            "feature": float(loss_feature.detach().cpu()),
            "z": float(loss_z.detach().cpu()),
            "teacher": float(loss_teacher.detach().cpu()),
            "raw": float(loss_raw_mse.detach().cpu()),
            "event_bce": float(loss_event_bce.detach().cpu()),
            "event_amp": float(loss_amp.detach().cpu()),
            "rate": float(loss_rate.detach().cpu()),
            "next": float(loss_next.detach().cpu()),
            "next_raw": float(loss_next_raw.detach().cpu()),
            "dx": float(loss_dx.detach().cpu()),
            "position": float(loss_pos.detach().cpu()),
            "local": float(loss_local.detach().cpu()),
            "popcorr": float(loss_popcorr.detach().cpu()),
            "coact": float(loss_coact.detach().cpu()),
        }

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, cfg_int(cfg, "epochs", 300) + 1):
        model.train()
        train_loss = 0.0
        nb = 0
        part_sum = {
            "feature": 0.0,
            "z": 0.0,
            "teacher": 0.0,
            "raw": 0.0,
            "event_bce": 0.0,
            "event_amp": 0.0,
            "rate": 0.0,
            "next": 0.0,
            "next_raw": 0.0,
            "dx": 0.0,
            "position": 0.0,
            "local": 0.0,
            "popcorr": 0.0,
            "coact": 0.0,
        }
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_on_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_float(cfg, "grad_clip", 5.0))
            opt.step()
            train_loss += float(loss.detach().cpu())
            for k, v in parts.items():
                part_sum[k] += v
            nb += 1
        train_loss /= max(1, nb)
        parts_avg = {k: v / max(1, nb) for k, v in part_sum.items()}

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
                f"feature {parts_avg['feature']:.5f} | z {parts_avg['z']:.5f} | "
                f"teacher {parts_avg['teacher']:.5f} | raw {parts_avg['raw']:.5f} | "
                f"event_bce {parts_avg['event_bce']:.5f} | rate {parts_avg['rate']:.5f} | "
                f"next {parts_avg['next']:.5f} | dx {parts_avg['dx']:.5f} | "
                f"pos {parts_avg['position']:.5f} | local {parts_avg['local']:.5f}"
            )

        if val_loss < best_val - cfg_float(cfg, "early_stopping_min_delta", 1e-5):
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if cfg_int(cfg, "early_stopping_patience", 50) > 0 and stale >= cfg_int(cfg, "early_stopping_patience", 50):
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} val {best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "position_geometry": position_meta,
        "local_lle_geometry": local_meta,
        "coactivity": coactivity_meta,
        "event_rate_mean": float(np.mean(event_rate)),
        "event_rate_median": float(np.median(event_rate)),
        "event_balanced_sampling": cfg_bool(cfg, "event_balanced_sampling", True),
        "loss_lambdas": {
            "feature": float(lambda_feature),
            "teacher_z": float(lambda_z),
            "teacher_recon": float(lambda_teacher),
            "raw_sparse": float(lambda_raw),
            "next": float(lambda_next),
            "next_raw": float(lambda_next_raw),
            "dx": float(lambda_dx),
            "position": float(lambda_pos),
            "local": float(lambda_local),
            "event_rate": float(lambda_rate),
            "coactivity": float(lambda_coact),
            "population_corr": float(lambda_popcorr),
            "event_bce_inside_raw": float(lambda_event_bce),
            "event_amplitude_inside_raw": float(lambda_event_amp),
        },
    }


def predict_model_outputs(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    fhat_chunks = []
    z_chunks = []
    fnext_chunks = []
    xhat_chunks = []
    prob_chunks = []
    amp_chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start:start + batch_size].astype(np.float32)).to(device)
            fhat, z, fnext, xhat, logits, amp = model(batch)
            fhat_chunks.append(fhat.cpu().numpy())
            z_chunks.append(z.cpu().numpy())
            fnext_chunks.append(fnext.cpu().numpy())
            xhat_chunks.append(xhat.cpu().numpy())
            prob_chunks.append(torch.sigmoid(logits).cpu().numpy())
            amp_chunks.append(amp.cpu().numpy())
    fhat_all = np.concatenate(fhat_chunks, axis=0).astype(np.float32)
    z_all = np.concatenate(z_chunks, axis=0).astype(np.float32)
    fnext_all = np.concatenate(fnext_chunks, axis=0).astype(np.float32)
    xhat_all = np.concatenate(xhat_chunks, axis=0).astype(np.float32)
    prob_all = np.concatenate(prob_chunks, axis=0).astype(np.float32)
    amp_all = np.concatenate(amp_chunks, axis=0).astype(np.float32)
    return fhat_all, z_all, fnext_all, xhat_all, prob_all, amp_all


def next_prediction_metrics(
    fnext_pred: np.ndarray | None,
    f_true: np.ndarray,
    x_true: np.ndarray,
    frame_trial_ids: np.ndarray,
    pca: PCA | None,
    norm_mu: np.ndarray | None,
    norm_sd: np.ndarray | None,
) -> Dict[str, Any]:
    if fnext_pred is None:
        return {}
    delta_f, valid = boundary_deltas(f_true, frame_trial_ids)
    delta_x, _ = boundary_deltas(x_true, frame_trial_ids)
    if not np.any(valid):
        return {"n_valid_next": 0}
    f_next_true = f_true + delta_f
    x_next_true = x_true + delta_x
    x_next_pred = neural.inverse_features_numpy(fnext_pred, pca, norm_mu, norm_sd)
    feature_err = fnext_pred[valid] - f_next_true[valid]
    raw_metrics = mind.corr_and_r2(x_next_true[valid], x_next_pred[valid])
    return {
        "n_valid_next": int(valid.sum()),
        "next_feature_mse": float(np.mean(feature_err ** 2)),
        "next_raw_r": raw_metrics["r"],
        "next_raw_R2": raw_metrics["R2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v7_population_geometry_ae.txt")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config key=value")
    args = parser.parse_args()

    cfg = mind.load_config(args.config)
    apply_overrides(cfg, args.overrides)
    seed = cfg_int(cfg, "seed", 42)
    neural.set_torch_seed(seed)
    device = neural.device_from_config(cfg)
    print(f"Device: {device}")

    out_dir = Path(cfg.get("out_dir", "runs/v7_population_geometry_ae_manual")).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(cfg, seed)
    teacher = build_teacher(cfg, data)
    model, train_meta = train_model(cfg, data, teacher, device)

    fhat_train, zhat_train, fnext_train, xhat_train_flat_raw, pevent_train, amp_train = predict_model_outputs(
        model, data["f_train_flat"], device, cfg_int(cfg, "predict_batch_size", 4096)
    )
    fhat_test, zhat_test, fnext_test, xhat_test_flat_raw, pevent_test, amp_test = predict_model_outputs(
        model, data["f_test_flat"], device, cfg_int(cfg, "predict_batch_size", 4096)
    )
    xhat_feature_train_flat = neural.inverse_features_numpy(fhat_train, data["pca"], data["norm_mu"], data["norm_sd"])
    xhat_feature_test_flat = neural.inverse_features_numpy(fhat_test, data["pca"], data["norm_mu"], data["norm_sd"])
    xhat_train_flat = xhat_train_flat_raw.astype(np.float32)
    xhat_test_flat = xhat_test_flat_raw.astype(np.float32)
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

    raw_meta = data["raw_split_meta"]
    train_metrics = mind.corr_and_r2(data["x_train_proc"], xhat_train)
    train_metrics.update(mind.summarize_frame_trial_mind_r2(
        data["x_train_proc"], xhat_train, raw_meta["train_frame_trial_ids"], raw_meta["train_trial_ids"]
    ))
    test_metrics = mind.corr_and_r2(data["x_test_proc"], xhat_test)
    test_metrics.update(mind.summarize_frame_trial_mind_r2(
        data["x_test_proc"], xhat_test, raw_meta["test_frame_trial_ids"], raw_meta["test_trial_ids"]
    ))
    event_metrics = mind.compute_event_metrics(data["x_test_proc"], xhat_test, percentile=cfg_float(cfg, "event_metric_percentile", 99.0))
    stratified_metrics = stratified_reconstruction_metrics(
        data["x_test_proc"],
        xhat_test,
        raw_meta["test_frame_trial_ids"],
        percentile=cfg_float(cfg, "event_metric_percentile", 99.0),
    )
    behavior_metrics = behavior_preservation_metrics(
        data["f_train_flat"],
        data["f_test_flat"],
        fhat_test,
        data["train_covariates"],
        data["test_covariates"],
    )
    next_metrics = {
        "train": next_prediction_metrics(
            fnext_train,
            data["f_train_flat"],
            data["x_train_flat"],
            raw_meta["train_frame_trial_ids"],
            data["pca"],
            data["norm_mu"],
            data["norm_sd"],
        ),
        "heldout": next_prediction_metrics(
            fnext_test,
            data["f_test_flat"],
            data["x_test_flat"],
            raw_meta["test_frame_trial_ids"],
            data["pca"],
            data["norm_mu"],
            data["norm_sd"],
        ),
    }

    teacher_train_decoded = teacher["teacher_train_target"]
    teacher_test_decoded = teacher["teacher_test_target"]
    if teacher["decoder_target"] in {"raw", "neural", "activity"}:
        teacher_xhat_train_flat = teacher_train_decoded.astype(np.float32)
        teacher_xhat_test_flat = teacher_test_decoded.astype(np.float32)
    else:
        teacher_xhat_train_flat = neural.inverse_features_numpy(teacher_train_decoded, data["pca"], data["norm_mu"], data["norm_sd"])
        teacher_xhat_test_flat = neural.inverse_features_numpy(teacher_test_decoded, data["pca"], data["norm_mu"], data["norm_sd"])
    teacher_xhat_train_flat, teacher_xhat_test_flat, teacher_post = mind.apply_reconstruction_postprocess(
        teacher_xhat_train_flat,
        teacher_xhat_test_flat,
        data["x_train_flat"],
        cfg,
    )
    teacher_xhat_train = teacher_xhat_train_flat.reshape(x_train_shape)
    teacher_xhat_test = teacher_xhat_test_flat.reshape(x_test_shape)
    teacher_test_metrics = mind.corr_and_r2(data["x_test_proc"], teacher_xhat_test)
    teacher_test_metrics.update(mind.summarize_frame_trial_mind_r2(
        data["x_test_proc"], teacher_xhat_test, raw_meta["test_frame_trial_ids"], raw_meta["test_trial_ids"]
    ))

    print(f"Population geometry AE train r {train_metrics['r']:.4f} | R2 {train_metrics['R2']:.4f} | MIND_R2 {train_metrics['MIND_R2']:.4f}")
    print(
        f"Population geometry AE heldout r {test_metrics['r']:.4f} | R2 {test_metrics['R2']:.4f} | "
        f"MIND_R2 {test_metrics['MIND_R2']:.4f} | trial median {test_metrics['MIND_R2_trial_median']:.4f}"
    )
    print(
        f"Teacher raw MIND heldout r {teacher_test_metrics['r']:.4f} | R2 {teacher_test_metrics['R2']:.4f} | "
        f"MIND_R2 {teacher_test_metrics['MIND_R2']:.4f} | trial median {teacher_test_metrics['MIND_R2_trial_median']:.4f}"
    )
    if next_metrics["heldout"]:
        print(
            f"Next-frame heldout feature MSE {next_metrics['heldout']['next_feature_mse']:.5f} | "
            f"raw next R2 {next_metrics['heldout']['next_raw_R2']:.4f}"
        )

    torch.save({"model_state": model.state_dict(), "config": jsonable(cfg), "train_meta": jsonable(train_meta)}, out_dir / "model.pt")
    np.savez_compressed(
        out_dir / "analysis_cache_best.npz",
        x_true=data["x_test_proc"].astype(np.float32),
        x_pred=xhat_test.astype(np.float32),
        x_train_true=data["x_train_proc"].astype(np.float32),
        x_train_pred=xhat_train.astype(np.float32),
        x_feature_pred=xhat_feature_test_flat.reshape(x_test_shape).astype(np.float32),
        x_train_feature_pred=xhat_feature_train_flat.reshape(x_train_shape).astype(np.float32),
        event_probability_test=pevent_test.reshape(x_test_shape).astype(np.float32),
        event_amplitude_test=amp_test.reshape(x_test_shape).astype(np.float32),
        z_test=zhat_test.reshape(x_test_shape[0], x_test_shape[1], -1).astype(np.float32),
        z_train=zhat_train.reshape(x_train_shape[0], x_train_shape[1], -1).astype(np.float32),
        z_teacher_train=teacher["z_train"].reshape(x_train_shape[0], x_train_shape[1], -1).astype(np.float32),
        z_teacher_test=teacher["z_test"].reshape(x_test_shape[0], x_test_shape[1], -1).astype(np.float32),
        train_trial_ids=raw_meta["train_trial_ids"],
        test_trial_ids=raw_meta["test_trial_ids"],
        train_frame_trial_ids=raw_meta["train_frame_trial_ids"],
        test_frame_trial_ids=raw_meta["test_frame_trial_ids"],
        train_frame_time=raw_meta["train_frame_time"],
        test_frame_time=raw_meta["test_frame_time"],
        train_frame_position=raw_meta["train_frame_position"],
        test_frame_position=raw_meta["test_frame_position"],
        metrics_json=np.asarray(json.dumps(jsonable({
            "train": train_metrics,
            "heldout": test_metrics,
            "events": event_metrics,
            "stratified": stratified_metrics,
            "behavior": behavior_metrics,
        }))),
    )
    save_raw_frame_recon_heatmap(
        data["x_test_proc"],
        xhat_test,
        out_dir / "raw_vs_recon_t0_15_n0_40.png",
        "Population geometry AE held-out reconstruction",
        frame_trial_ids=raw_meta["test_frame_trial_ids"],
        frame_time=raw_meta["test_frame_time"],
    )
    mind.save_embedding_plot(teacher["z_lm"], out_dir / "latent_manifold_mds.png")

    final_metrics = {
        "train": train_metrics,
        "heldout": test_metrics,
        "events": event_metrics,
        "stratified": stratified_metrics,
        "behavior_preservation": behavior_metrics,
        "next_prediction": next_metrics,
        "teacher_heldout": teacher_test_metrics,
        "R2": test_metrics["R2"],
        "r": test_metrics["r"],
    }
    torch.save(final_metrics, out_dir / "final_metrics.pt")
    metadata = {
        "script": Path(__file__).name,
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(data["data_path"]),
        "data_mode": data["data_mode"],
        "assumptions": [
            "raw observed neural frames",
            "whole-trial heldout split",
            "no position bins",
            "no position/behavior covariates in model input",
            "weak same-trial position triplets as metadata-only loss",
            "sparse event raw observation head",
            "event-rate penalty to reduce overfiring",
            "local LLE population geometry loss from train neural states",
            "coactivity and population correlation losses",
            "trial-boundary-aware transitions",
            "one-step transition head predicts next PCA feature",
        ],
        "n_trials_total": int(raw_meta["n_trials_total"]),
        "n_trials_train": int(raw_meta["n_trials_train"]),
        "n_trials_heldout": int(raw_meta["n_trials_heldout"]),
        "heldout_fraction": float(raw_meta["heldout_fraction"]),
        "n_train_frames": int(raw_meta["n_train_frames"]),
        "n_heldout_frames": int(raw_meta["n_test_frames"]),
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
        "raw_frame_split": raw_meta,
        "training": train_meta,
        "metrics": final_metrics,
        "feature_decoder_raw_metrics": {
            "train": mind.corr_and_r2(data["x_train_proc"], xhat_feature_train_flat.reshape(x_train_shape)),
            "heldout": mind.corr_and_r2(data["x_test_proc"], xhat_feature_test_flat.reshape(x_test_shape)),
        },
        "method_note": "Raw-frame population-geometry AE: PCA(raw neural frame) -> z -> PCA feature, sparse event raw decoder z -> xhat, z -> next PCA feature, train-only neural-state geometry, no position/behavior input.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2) + "\n")
    (out_dir / "training_results.txt").write_text(json.dumps(jsonable(final_metrics), indent=2) + "\n")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
