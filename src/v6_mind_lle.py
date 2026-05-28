#!/usr/bin/env python3
"""MIND-style manifold/LLE pipeline for neural population trials.

This is intentionally separate from the neural VAE/ODE training code. It builds a
train-only population-state manifold, embeds transition-derived distances, and
uses LLE for out-of-sample mapping/reconstruction on held-out trials.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = parse_value(value)
    return cfg


def resolve_path(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            REPO_ROOT / p,
            Path(__file__).resolve().parent / p,
            REPO_ROOT / "src" / p,
            REPO_ROOT / "src" / "npz_e65_data" / p.name,
        ])
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def get_npz_array(data: np.lib.npyio.NpzFile, names: Iterable[str]) -> np.ndarray | None:
    lower = {k.lower(): k for k in data.files}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return data[key]
    return None


def load_neural_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    roi = get_npz_array(data, ["roi", "ROI", "ROIactivities", "activity", "X"])
    trials = get_npz_array(data, ["Trial", "trial", "trials"])
    time = get_npz_array(data, ["Time", "time", "times"])
    position = get_npz_array(data, ["Position", "position", "pos", "linear_position"])
    if roi is None or trials is None:
        raise KeyError(f"Could not find roi/trial arrays in {path}; keys={data.files}")
    roi = np.asarray(roi, dtype=np.float32)
    trials = np.asarray(trials).reshape(-1)
    if roi.shape[0] != trials.shape[0] and roi.shape[1] == trials.shape[0]:
        roi = roi.T
    if roi.shape[0] != trials.shape[0]:
        raise ValueError(f"ROI shape {roi.shape} does not align with Trial shape {trials.shape}")
    if time is not None:
        time = np.asarray(time, dtype=np.float32).reshape(-1)
    if position is not None:
        position = np.asarray(position, dtype=np.float32).reshape(-1)
    return roi, trials, time, position


def mind_active_neuron_mask(roi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    # MIND-style filter: keep neurons with any positive activity over the full run.
    return np.sum(roi, axis=0) > threshold


def group_indices_by_trial(trials: np.ndarray, drop_first_trials: int = 0, min_frames: int = 2) -> Tuple[np.ndarray, list[np.ndarray]]:
    unique = np.array(sorted(np.unique(trials)))
    if drop_first_trials > 0:
        unique = unique[drop_first_trials:]
    trial_ids = []
    groups = []
    for tid in unique:
        idx = np.flatnonzero(trials == tid)
        if idx.size >= min_frames:
            trial_ids.append(tid)
            groups.append(idx)
    return np.asarray(trial_ids), groups


def average_repeated_axis(axis: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(axis, kind="mergesort")
    axis_sorted = np.asarray(axis[order], dtype=np.float64)
    values_sorted = np.asarray(values[order], dtype=np.float32)
    uniq, inverse, counts = np.unique(axis_sorted, return_inverse=True, return_counts=True)
    if uniq.size == axis_sorted.size:
        return axis_sorted, values_sorted
    sums = np.zeros((uniq.size, values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values_sorted)
    avg = (sums / counts[:, None]).astype(np.float32)
    return uniq, avg


def resample_sequence(values: np.ndarray, axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    axis, values = average_repeated_axis(axis, values)
    if axis.size == 0:
        raise ValueError("Cannot resample an empty trial")
    if axis.size == 1 or np.allclose(axis[0], axis[-1]):
        return np.repeat(values[:1], target_axis.size, axis=0)
    out = np.empty((target_axis.size, values.shape[1]), dtype=np.float32)
    for n in range(values.shape[1]):
        out[:, n] = np.interp(target_axis, axis, values[:, n]).astype(np.float32)
    return out


def build_trial_sequences(
    roi: np.ndarray,
    trials: np.ndarray,
    time: np.ndarray | None,
    position: np.ndarray | None,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps = float(cfg.get("fps", 10.0))
    trial_len_s = float(cfg.get("trial_len_s", 12.0))
    seq_len = int(round(fps * trial_len_s))
    if seq_len < 2:
        raise ValueError("sequence length must be at least 2")
    drop_first = int(cfg.get("drop_first_trials", 0))
    min_frames = int(cfg.get("min_frames", 2))
    align = str(cfg.get("sequence_alignment", cfg.get("time_mode", "time"))).lower()
    trial_ids, groups = group_indices_by_trial(trials, drop_first, min_frames)
    if len(groups) == 0:
        raise ValueError("No usable trials after filtering")

    global_position_range: tuple[float, float] | None = None
    if align in {"position", "pos", "maze_position"} and position is not None and bool(cfg.get("position_normalize", True)):
        configured_lo = float(cfg.get("position_min", np.nan))
        configured_hi = float(cfg.get("position_max", np.nan))
        if np.isfinite(configured_lo) and np.isfinite(configured_hi):
            global_position_range = (configured_lo, configured_hi)
        else:
            all_pos = []
            for idx in groups:
                pos = np.asarray(position[idx], dtype=np.float64)
                if bool(cfg.get("position_cumulative_max", True)):
                    pos = np.maximum.accumulate(pos)
                all_pos.append(pos)
            merged = np.concatenate(all_pos)
            global_position_range = (float(np.nanmin(merged)), float(np.nanmax(merged)))

    sequences = []
    for idx in groups:
        x = roi[idx]
        if align in {"position", "pos", "maze_position"}:
            if position is None:
                raise KeyError("sequence_alignment=position requires a Position array in the NPZ")
            axis = np.asarray(position[idx], dtype=np.float64)
            if bool(cfg.get("position_cumulative_max", True)):
                axis = np.maximum.accumulate(axis)
            if bool(cfg.get("position_normalize", True)):
                lo, hi = global_position_range if global_position_range is not None else (float(np.nanmin(axis)), float(np.nanmax(axis)))
                if hi <= lo:
                    target = np.linspace(axis[0], axis[-1], seq_len)
                else:
                    axis = np.clip((axis - lo) / (hi - lo), 0.0, 1.0)
                    target = np.linspace(0.0, 1.0, seq_len)
            else:
                target = np.linspace(float(np.nanmin(axis)), float(np.nanmax(axis)), seq_len)
        else:
            if time is not None:
                axis = np.asarray(time[idx], dtype=np.float64)
                axis = axis - axis[0]
            else:
                axis = np.arange(idx.size, dtype=np.float64) / fps
            if bool(cfg.get("normalize_time", True)):
                target = np.linspace(axis[0], axis[-1], seq_len)
            else:
                target = np.linspace(0.0, trial_len_s, seq_len)
        sequences.append(resample_sequence(x, axis, target))
    return np.stack(sequences, axis=0), trial_ids, np.linspace(0.0, 1.0, seq_len, dtype=np.float32)


def baseline_correct_trials(x: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    base = np.quantile(x, quantile, axis=1, keepdims=True)
    return (x - base).astype(np.float32)


def split_trials(n_trials: int, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random(n_trials) >= frac
    if mask.all():
        mask[rng.integers(0, n_trials)] = False
    if (~mask).all():
        mask[rng.integers(0, n_trials)] = True
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def flatten_trials(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def sequence_deltas(features: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(features)
    deltas[:, :-1, :] = features[:, 1:, :] - features[:, :-1, :]
    if features.shape[1] > 1:
        deltas[:, -1, :] = deltas[:, -2, :]
    return deltas


def farthest_point_landmarks(
    x: np.ndarray,
    count: int,
    rng: np.random.Generator,
    initial_indices: np.ndarray | None = None,
) -> np.ndarray:
    n = x.shape[0]
    count = min(max(int(count), 1), n)
    if initial_indices is None or len(initial_indices) == 0:
        selected = np.empty(count, dtype=np.int64)
        selected[0] = int(rng.integers(0, n))
        start = 1
    else:
        unique_initial = np.unique(np.asarray(initial_indices, dtype=np.int64))
        unique_initial = unique_initial[(unique_initial >= 0) & (unique_initial < n)]
        selected = np.empty(count, dtype=np.int64)
        selected[: min(count, unique_initial.size)] = unique_initial[:count]
        start = min(count, unique_initial.size)
        if start == 0:
            selected[0] = int(rng.integers(0, n))
            start = 1
    min_dist = np.full(n, np.inf, dtype=np.float32)
    for idx in selected[:start]:
        diff = x - x[int(idx)]
        dist = np.einsum("ij,ij->i", diff, diff)
        min_dist = np.minimum(min_dist, dist)
    min_dist[selected[:start]] = -np.inf
    for i in range(start, count):
        idx = int(np.argmax(min_dist))
        selected[i] = idx
        diff = x - x[idx]
        dist = np.einsum("ij,ij->i", diff, diff)
        min_dist = np.minimum(min_dist, dist)
        min_dist[idx] = -np.inf
    return selected


def score_activity(x: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode in {"max", "peak"}:
        return np.max(x, axis=1)
    if mode in {"sum", "population"}:
        return np.sum(x, axis=1)
    if mode in {"l2", "norm"}:
        return np.linalg.norm(x, axis=1)
    if mode in {"mean"}:
        return np.mean(x, axis=1)
    raise ValueError(f"Unknown landmark_activity_score={mode!r}")


def score_transition(dx: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    abs_dx = np.abs(dx)
    if mode in {"max", "max_abs", "peak"}:
        return np.max(abs_dx, axis=1)
    if mode in {"sum", "population"}:
        return np.sum(abs_dx, axis=1)
    if mode in {"l2", "norm"}:
        return np.linalg.norm(dx, axis=1)
    if mode in {"mean"}:
        return np.mean(abs_dx, axis=1)
    raise ValueError(f"Unknown landmark_transition_score={mode!r}")


def take_top_diverse(
    score: np.ndarray,
    features: np.ndarray,
    count: int,
    used: set[int],
    rng: np.random.Generator,
    pool_multiplier: float,
) -> list[int]:
    if count <= 0:
        return []
    pool_size = min(score.size, max(count, int(math.ceil(count * pool_multiplier))))
    pool: list[int] = []
    for idx in np.argsort(score)[::-1]:
        i = int(idx)
        if i in used:
            continue
        pool.append(i)
        if len(pool) >= pool_size:
            break
    if len(pool) <= count:
        chosen = pool
    else:
        pool_arr = np.asarray(pool, dtype=np.int64)
        local = farthest_point_landmarks(features[pool_arr], count, rng, initial_indices=np.array([0]))
        chosen = [int(pool_arr[i]) for i in local]
    for i in chosen:
        used.add(int(i))
    return chosen


def select_landmarks(
    features: np.ndarray,
    raw_states: np.ndarray,
    raw_deltas: np.ndarray,
    count: int,
    rng: np.random.Generator,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    count = min(max(int(count), 1), features.shape[0])
    if not bool(cfg.get("event_aware_landmarks", False)):
        idx = farthest_point_landmarks(features, count, rng)
        return idx, {
            "mode": "coverage_only",
            "requested": int(count),
            "activity_count": 0,
            "transition_count": 0,
            "coverage_count": int(idx.size),
        }

    activity_fraction = float(cfg.get("landmark_activity_fraction", 0.15))
    transition_fraction = float(cfg.get("landmark_transition_fraction", 0.15))
    pool_multiplier = float(cfg.get("landmark_event_pool_multiplier", 5.0))
    activity_count = int(round(count * activity_fraction))
    transition_count = int(round(count * transition_fraction))
    activity_count = min(max(activity_count, 0), count)
    transition_count = min(max(transition_count, 0), count - activity_count)

    used: set[int] = set()
    activity = score_activity(raw_states, str(cfg.get("landmark_activity_score", "max")))
    transition = score_transition(raw_deltas, str(cfg.get("landmark_transition_score", "max_abs")))
    activity_idx = take_top_diverse(activity, features, activity_count, used, rng, pool_multiplier)
    transition_idx = take_top_diverse(transition, features, transition_count, used, rng, pool_multiplier)
    seeded = np.asarray(activity_idx + transition_idx, dtype=np.int64)
    idx = farthest_point_landmarks(features, count, rng, initial_indices=seeded)
    seeded_set = set(int(i) for i in seeded)
    meta = {
        "mode": "event_aware",
        "requested": int(count),
        "activity_count": int(len(activity_idx)),
        "transition_count": int(len(transition_idx)),
        "coverage_count": int(count - len(seeded_set)),
        "activity_fraction": float(activity_fraction),
        "transition_fraction": float(transition_fraction),
        "event_pool_multiplier": float(pool_multiplier),
        "activity_score": str(cfg.get("landmark_activity_score", "max")),
        "transition_score": str(cfg.get("landmark_transition_score", "max_abs")),
        "activity_score_min_selected": float(np.min(activity[activity_idx])) if activity_idx else 0.0,
        "transition_score_min_selected": float(np.min(transition[transition_idx])) if transition_idx else 0.0,
        "activity_score_max": float(np.max(activity)) if activity.size else 0.0,
        "transition_score_max": float(np.max(transition)) if transition.size else 0.0,
    }
    return idx, meta


def standardize_features(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def transition_geodesic_distances(
    features: np.ndarray,
    deltas: np.ndarray,
    k: int,
    transition_weight: float,
    temperature: float,
    sym: str = "min",
    probability_mode: str = "next_state",
    exclude_self: bool = True,
    use_graph_geodesics: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = features.shape[0]
    k = min(max(int(k), 2), max(2, n - 1))
    raw_phi = np.concatenate([features, float(transition_weight) * deltas], axis=1)
    phi, phi_mu, phi_sd = standardize_features(raw_phi)
    probability_mode = probability_mode.lower()
    if probability_mode in {"next", "next_state", "transition"}:
        raw_query = np.concatenate([features + deltas, float(transition_weight) * deltas], axis=1)
        query = ((raw_query - phi_mu) / phi_sd).astype(np.float32)
    elif probability_mode in {"local", "phi", "state_delta"}:
        query = phi
    else:
        raise ValueError(f"Unknown transition_probability_mode={probability_mode!r}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(phi)
    distances_full, indices_full = nn.kneighbors(query, return_distance=True)
    distances = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row_d = []
        row_i = []
        for dist, idx in zip(distances_full[i], indices_full[i]):
            if exclude_self and int(idx) == i:
                continue
            row_d.append(float(dist))
            row_i.append(int(idx))
            if len(row_i) >= k:
                break
        if len(row_i) < k:
            for dist, idx in zip(distances_full[i], indices_full[i]):
                row_d.append(float(dist))
                row_i.append(int(idx))
                if len(row_i) >= k:
                    break
        distances[i] = np.asarray(row_d[:k], dtype=np.float32)
        indices[i] = np.asarray(row_i[:k], dtype=np.int64)

    scales = np.median(distances, axis=1, keepdims=True)
    scales = np.where(scales <= 1e-8, np.mean(distances, axis=1, keepdims=True) + 1e-8, scales)
    scaled = distances / scales
    logits = -(scaled ** 2) / max(float(temperature), 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    edge_weights = np.sqrt(-np.log(np.clip(probs, 1e-12, 1.0))).astype(np.float32)

    if not use_graph_geodesics:
        diff = phi[:, None, :] - phi[None, :, :]
        d_global = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).astype(np.float32)
        np.fill_diagonal(d_global, 0.0)
        meta = {
            "transition_k": int(k),
            "transition_weight": float(transition_weight),
            "transition_temperature": float(temperature),
            "transition_probability_mode": probability_mode,
            "transition_feature": "phi_t=[x_t, alpha * (x_t+1 - x_t)]",
            "transition_query": "phi_next=[x_t + dx_t, alpha * dx_t]",
            "use_graph_geodesics": False,
            "disconnected_pairs_replaced": 0,
            "distance_min_positive": float(np.min(d_global[d_global > 0])) if np.any(d_global > 0) else 0.0,
            "distance_max": float(np.max(d_global)),
        }
        return d_global, meta

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    vals = edge_weights.reshape(-1)
    edge = np.full((n, n), np.inf, dtype=np.float32)
    edge[rows, cols] = vals
    if sym == "avg":
        rev = edge.T
        finite_edge = np.isfinite(edge)
        finite_rev = np.isfinite(rev)
        both = finite_edge & finite_rev
        either = finite_edge | finite_rev
        sym_edge = np.full_like(edge, np.inf)
        sym_edge[both] = 0.5 * (edge[both] + rev[both])
        only_one = either & ~both
        sym_edge[only_one] = np.minimum(edge[only_one], rev[only_one])
    else:
        sym_edge = np.minimum(edge, edge.T)
    np.fill_diagonal(sym_edge, 0.0)
    graph = coo_matrix(np.where(np.isfinite(sym_edge), sym_edge, 0.0)).tocsr()
    d_geo = shortest_path(graph, directed=False, unweighted=False)
    finite = np.isfinite(d_geo)
    disconnected = int((~finite).sum())
    if disconnected:
        max_finite = float(np.max(d_geo[finite])) if finite.any() else 1.0
        d_geo = np.where(finite, d_geo, 2.0 * max_finite)
    np.fill_diagonal(d_geo, 0.0)
    meta = {
        "transition_k": int(k),
        "transition_weight": float(transition_weight),
        "transition_temperature": float(temperature),
        "transition_probability_mode": probability_mode,
        "transition_feature": "phi_t=[x_t, alpha * (x_t+1 - x_t)]",
        "transition_query": "phi_next=[x_t + dx_t, alpha * dx_t]",
        "use_graph_geodesics": True,
        "disconnected_pairs_replaced": disconnected,
        "distance_min_positive": float(np.min(d_geo[d_geo > 0])) if np.any(d_geo > 0) else 0.0,
        "distance_max": float(np.max(d_geo)),
    }
    return d_geo.astype(np.float32), meta


def classical_mds(d: np.ndarray, dim: int) -> np.ndarray:
    n = d.shape[0]
    d2 = d.astype(np.float64) ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[:dim], 0.0)
    z = vecs[:, :dim] * np.sqrt(pos + 1e-12)
    if z.shape[1] < dim:
        z = np.pad(z, ((0, 0), (0, dim - z.shape[1])))
    return z.astype(np.float32)


def sammon_refine(d: np.ndarray, z0: np.ndarray, iters: int, lr: float, seed: int) -> Tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and bool(int(os.environ.get("USE_CUDA_FOR_SAMMON", "0"))) else "cpu")
    d_t = torch.as_tensor(d, dtype=torch.float32, device=device)
    mask = d_t > 1e-8
    weights = torch.where(mask, 1.0 / (d_t + 1e-6), torch.zeros_like(d_t))
    norm = torch.clamp(weights.sum(), min=1.0)
    z = torch.nn.Parameter(torch.as_tensor(z0, dtype=torch.float32, device=device).clone())
    opt = torch.optim.Adam([z], lr=float(lr))
    last = 0.0
    for _ in range(max(int(iters), 0)):
        opt.zero_grad(set_to_none=True)
        dz = torch.cdist(z, z)
        loss = (weights * (dz - d_t).pow(2)).sum() / norm
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    return z.detach().cpu().numpy().astype(np.float32), last


def embed_distances(d: np.ndarray, dim: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    z0 = classical_mds(d, dim)
    mode = str(cfg.get("embedding_mode", "sammon")).lower()
    meta = {"embedding_mode": mode}
    if mode in {"sammon", "mds_sammon"}:
        z, loss = sammon_refine(
            d,
            z0,
            iters=int(cfg.get("sammon_iters", 500)),
            lr=float(cfg.get("sammon_lr", 0.01)),
            seed=int(cfg.get("seed", cfg.get("mind_split_seed", 42))),
        )
        meta["sammon_final_loss"] = float(loss)
        return z, meta
    return z0, meta


class LLEMapper:
    def __init__(self, n_neighbors: int = 16, ridge: float = 1e-3):
        self.n_neighbors = int(n_neighbors)
        self.ridge = float(ridge)
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LLEMapper":
        if x.shape[0] != y.shape[0]:
            raise ValueError("LLE fit arrays must have the same number of rows")
        self.x_train = np.asarray(x, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.float32)
        k = min(self.n_neighbors, self.x_train.shape[0])
        self.nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self.nn.fit(self.x_train)
        return self

    def _weights(self, xi: np.ndarray, nb: np.ndarray) -> np.ndarray:
        centered = nb - xi[None, :]
        gram = centered @ centered.T
        trace = float(np.trace(gram))
        reg = self.ridge * (trace / max(gram.shape[0], 1) if trace > 0 else 1.0)
        gram = gram + np.eye(gram.shape[0], dtype=np.float32) * (reg + 1e-8)
        ones = np.ones(gram.shape[0], dtype=np.float32)
        try:
            w = np.linalg.solve(gram, ones)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(gram, ones, rcond=None)[0]
        s = float(np.sum(w))
        if abs(s) < 1e-12:
            return np.full_like(w, 1.0 / w.size)
        return (w / s).astype(np.float32)

    def transform(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if self.x_train is None or self.y_train is None or self.nn is None:
            raise RuntimeError("LLEMapper must be fit before transform")
        x = np.asarray(x, dtype=np.float32)
        out = np.empty((x.shape[0], self.y_train.shape[1]), dtype=np.float32)
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            inds = self.nn.kneighbors(x[start:end], return_distance=False)
            for r, row in enumerate(inds):
                xi = x[start + r]
                w = self._weights(xi, self.x_train[row])
                out[start + r] = w @ self.y_train[row]
        return out


class KernelRegressor:
    """Nonnegative local weighted-average map.

    This is safer than inverse LLE for held-out points because it interpolates
    among nearby train states instead of solving unconstrained affine weights.
    """

    def __init__(self, n_neighbors: int = 32, bandwidth: float = 1.0, eps: float = 1e-8):
        self.n_neighbors = int(n_neighbors)
        self.bandwidth = float(bandwidth)
        self.eps = float(eps)
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRegressor":
        if x.shape[0] != y.shape[0]:
            raise ValueError("Kernel fit arrays must have the same number of rows")
        self.x_train = np.asarray(x, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.float32)
        k = min(max(self.n_neighbors, 1), self.x_train.shape[0])
        self.nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self.nn.fit(self.x_train)
        return self

    def transform(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if self.x_train is None or self.y_train is None or self.nn is None:
            raise RuntimeError("KernelRegressor must be fit before transform")
        x = np.asarray(x, dtype=np.float32)
        out = np.empty((x.shape[0], self.y_train.shape[1]), dtype=np.float32)
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            dists, inds = self.nn.kneighbors(x[start:end], return_distance=True)
            scale = np.median(dists, axis=1, keepdims=True)
            fallback = np.maximum(dists[:, -1:], self.eps)
            scale = np.where(scale > self.eps, scale, fallback)
            denom = np.maximum(scale * max(self.bandwidth, self.eps), self.eps)
            logits = -0.5 * (dists / denom) ** 2
            logits = logits - logits.max(axis=1, keepdims=True)
            weights = np.exp(logits).astype(np.float32)
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), self.eps)
            for r, row in enumerate(inds):
                out[start + r] = weights[r] @ self.y_train[row]
        return out


def corr_and_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return {"r": float("nan"), "R2": float("nan"), "variance_explained": float("nan")}
    r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    sse = float(np.sum((a - b) ** 2))
    sst = float(np.sum((a - np.mean(a)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    var_exp = 1.0 - float(np.var(a - b)) / float(np.var(a)) if np.var(a) > 0 else float("nan")
    return {"r": r, "R2": float(r2), "variance_explained": float(var_exp)}


def r2_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    sst = float(np.sum((a - np.mean(a)) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / sst)


def apply_reconstruction_postprocess(
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    train_target: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"enabled": bool(cfg.get("postprocess_reconstruction", False))}
    if not meta["enabled"]:
        return train_pred, test_pred, meta

    train_out = train_pred.astype(np.float32, copy=True)
    test_out = test_pred.astype(np.float32, copy=True)
    train_before = r2_value(train_target, train_out)
    meta["train_R2_before_postprocess"] = train_before

    clip_min = cfg.get("recon_clip_min", None)
    if clip_min is not None:
        clip_min = float(clip_min)
        train_out = np.maximum(train_out, clip_min)
        test_out = np.maximum(test_out, clip_min)
        meta["clip_min"] = clip_min

    clip_max = cfg.get("recon_clip_max", None)
    if clip_max is not None:
        clip_max = float(clip_max)
        train_out = np.minimum(train_out, clip_max)
        test_out = np.minimum(test_out, clip_max)
        meta["clip_max"] = clip_max

    floor_mode = str(cfg.get("recon_floor_mode", "none")).lower()
    floor = 0.0
    if floor_mode in {"fixed", "hard", "threshold"}:
        floor = float(cfg.get("recon_activity_floor", 0.0))
    elif floor_mode in {"train_opt", "opt", "optimize", "train"}:
        max_floor_cfg = cfg.get("recon_floor_max", None)
        if max_floor_cfg is None:
            max_floor = float(np.nanpercentile(train_out, 99.0))
        else:
            max_floor = float(max_floor_cfg)
        steps = max(int(cfg.get("recon_floor_steps", 61)), 2)
        candidates = np.linspace(0.0, max_floor, steps, dtype=np.float32)
        best_floor = 0.0
        best_r2 = -np.inf
        for cand in candidates:
            candidate_pred = np.where(train_out >= cand, train_out, 0.0)
            score = r2_value(train_target, candidate_pred)
            if np.isfinite(score) and score > best_r2:
                best_r2 = score
                best_floor = float(cand)
        floor = best_floor
        meta["train_optimized_floor_R2"] = float(best_r2)
    elif floor_mode not in {"none", "off", "false"}:
        raise ValueError(f"Unknown recon_floor_mode={floor_mode!r}")

    if floor > 0:
        train_out = np.where(train_out >= floor, train_out, 0.0).astype(np.float32)
        test_out = np.where(test_out >= floor, test_out, 0.0).astype(np.float32)
    meta["recon_floor_mode"] = floor_mode
    meta["recon_activity_floor"] = float(floor)
    meta["train_R2_after_postprocess"] = r2_value(train_target, train_out)
    return train_out, test_out, meta


def compute_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 99.0) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)
    abs_true = np.abs(true_flat)
    out: Dict[str, float] = {}
    for pct, name in [(99.0, "top_1_percent_event_capture"), (99.5, "top_0_5_percent_event_capture")]:
        thr = np.percentile(abs_true, pct)
        mask = abs_true >= thr
        denom = np.sum(np.abs(true_flat[mask]))
        out[name] = float(np.sum(np.abs(pred_flat[mask])) / denom) if denom > 0 else float("nan")
    out["pred_std_over_true_std"] = float(np.std(pred_flat) / (np.std(true_flat) + 1e-12))
    true_dx = np.diff(true, axis=1).reshape(-1)
    pred_dx = np.diff(pred, axis=1).reshape(-1)
    out["pred_dynamics_std_over_true_dynamics_std"] = float(np.std(pred_dx) / (np.std(true_dx) + 1e-12))
    per = []
    for n in range(true.shape[-1]):
        t = true[..., n].reshape(-1)
        p = pred[..., n].reshape(-1)
        sst = np.sum((t - np.mean(t)) ** 2)
        per.append(1.0 - np.sum((t - p) ** 2) / sst if sst > 0 else np.nan)
    per_arr = np.asarray(per, dtype=np.float64)
    out["per_neuron_R2_mean"] = float(np.nanmean(per_arr))
    out["per_neuron_R2_median"] = float(np.nanmedian(per_arr))
    out["per_neuron_R2_fraction_positive"] = float(np.mean(per_arr > 0))
    frame_score = np.max(np.abs(true), axis=-1)
    thr = np.percentile(frame_score.reshape(-1), float(percentile))
    mask = frame_score >= thr
    if np.any(mask):
        out.update({"event_frame_" + k: v for k, v in corr_and_r2(true[mask], pred[mask]).items()})
        out["event_frame_count"] = int(mask.sum())
        out["event_frame_fraction"] = float(mask.mean())
    return out


def save_recon_heatmap(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    t_end = min(16, y_true.shape[1])
    n_end = min(41, y_true.shape[2])
    raw = y_true[0, :t_end, :n_end].T
    recon = y_pred[0, :t_end, :n_end].T
    resid = raw - recon
    vmax = float(np.nanmax(np.abs(raw))) if raw.size else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, arr, label in zip(axes, [raw, recon, resid], ["raw", "recon", "raw - recon"]):
        im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis" if label != "raw - recon" else "coolwarm")
        ax.set_title(label)
        ax.set_xlabel("time bin")
        ax.set_ylabel("neuron")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_embedding_plot(z: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    if z.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        colors = np.linspace(0, 1, z.shape[0])
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=colors, cmap="viridis", s=8)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
    else:
        ax = fig.add_subplot(111)
        colors = np.linspace(0, 1, z.shape[0])
        y = z[:, 1] if z.shape[1] > 1 else np.zeros(z.shape[0])
        ax.scatter(z[:, 0], y, c=colors, cmap="viridis", s=8)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    ax.set_title("MIND-style landmark embedding")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", cfg.get("mind_split_seed", 42)))
    rng = set_seed(seed)
    out_dir = Path(cfg.get("out_dir", "runs/v6_mind_lle_manual")).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(cfg.get("data_path", "src/npz_e65_data/E65_data.npz"))
    print(f"Loading data: {data_path}")
    roi, trials, time, position = load_neural_data(data_path)
    print(f"Raw ROI: frames={roi.shape[0]}, neurons={roi.shape[1]}")

    if bool(cfg.get("filter_inactive_neurons", True)):
        mask = mind_active_neuron_mask(roi, float(cfg.get("inactive_neuron_threshold", 0.0)))
        removed = int((~mask).sum())
        roi = roi[:, mask]
        print(f"MIND neuron filter: removed {removed} globally silent neurons; kept {roi.shape[1]} neurons.")
    else:
        mask = np.ones(roi.shape[1], dtype=bool)
        removed = 0

    sequences, trial_ids, axis = build_trial_sequences(roi, trials, time, position, cfg)
    print(f"Built sequences: B={sequences.shape[0]}, L={sequences.shape[1]}, N={sequences.shape[2]}")

    train_idx, test_idx = split_trials(sequences.shape[0], float(cfg.get("test_frac", cfg.get("mind_test_frac", 0.1))), int(cfg.get("mind_split_seed", seed)))
    x_train = sequences[train_idx]
    x_test = sequences[test_idx]
    print(f"Trial split: train={len(train_idx)}, heldout={len(test_idx)} ({len(test_idx) / sequences.shape[0]:.3f})")

    if bool(cfg.get("filter_train_silent_neurons", False)):
        std = flatten_trials(x_train).std(axis=0)
        keep = std > float(cfg.get("train_silent_threshold", 1e-7))
        train_removed = int((~keep).sum())
        x_train = x_train[..., keep]
        x_test = x_test[..., keep]
        mask_indices = np.flatnonzero(mask)
        mask[mask_indices[~keep]] = False
        print(f"Train-std neuron filter: removed {train_removed}; kept {x_train.shape[-1]} neurons.")

    if bool(cfg.get("baseline_correct", False)):
        q = float(cfg.get("baseline_quantile", 0.1))
        x_train_proc = baseline_correct_trials(x_train, q)
        x_test_proc = baseline_correct_trials(x_test, q)
    else:
        x_train_proc = x_train.astype(np.float32)
        x_test_proc = x_test.astype(np.float32)

    x_train_flat = flatten_trials(x_train_proc)
    x_test_flat = flatten_trials(x_test_proc)

    norm_mu = None
    norm_sd = None
    if bool(cfg.get("normalize_inputs", False)):
        norm_mu = x_train_flat.mean(axis=0, keepdims=True).astype(np.float32)
        norm_sd = x_train_flat.std(axis=0, keepdims=True).astype(np.float32)
        norm_sd = np.where(norm_sd < float(cfg.get("normalize_eps", 1e-6)), 1.0, norm_sd)
        x_train_flat_proc = ((x_train_flat - norm_mu) / norm_sd).astype(np.float32)
        x_test_flat_proc = ((x_test_flat - norm_mu) / norm_sd).astype(np.float32)
    else:
        x_train_flat_proc = x_train_flat.astype(np.float32)
        x_test_flat_proc = x_test_flat.astype(np.float32)

    use_pca = bool(cfg.get("use_pca", False)) or int(cfg.get("pca_dim", 0) or 0) > 0
    pca = None
    if use_pca:
        pca_dim = int(cfg.get("pca_dim", 0) or 0)
        if pca_dim <= 0:
            pca_dim = min(x_train_flat_proc.shape[1], int(cfg.get("pca_max_dim", 64)))
        pca = PCA(n_components=min(pca_dim, x_train_flat_proc.shape[1]), random_state=seed)
        f_train_flat = pca.fit_transform(x_train_flat_proc).astype(np.float32)
        f_test_flat = pca.transform(x_test_flat_proc).astype(np.float32)
        pca_var = float(np.sum(pca.explained_variance_ratio_))
        print(f"PCA feature space: dim={f_train_flat.shape[1]}, explained_variance={pca_var:.4f}")
    else:
        f_train_flat = x_train_flat_proc
        f_test_flat = x_test_flat_proc
        pca_var = float("nan")
        print(f"Feature space: raw neurons, dim={f_train_flat.shape[1]}")

    b_train, seq_len, feat_dim = x_train.shape[0], x_train.shape[1], f_train_flat.shape[1]
    f_train_seq = f_train_flat.reshape(b_train, seq_len, feat_dim)
    delta_train_flat = flatten_trials(sequence_deltas(f_train_seq))
    raw_delta_train_flat = flatten_trials(sequence_deltas(x_train_proc))

    manifold_mask = np.ones(f_train_flat.shape[0], dtype=bool)
    manifold_meta: Dict[str, Any] = {
        "filter_silent_frames_for_manifold": bool(cfg.get("filter_silent_frames_for_manifold", False)),
        "total_train_states": int(f_train_flat.shape[0]),
    }
    if bool(cfg.get("filter_silent_frames_for_manifold", False)):
        score_mode = str(cfg.get("silent_frame_score", "sum"))
        frame_score = score_activity(x_train_flat, score_mode)
        threshold = float(cfg.get("silent_frame_threshold", 0.0))
        manifold_mask = frame_score > threshold
        if manifold_mask.sum() < max(2, int(cfg.get("transition_k", 16)) + 1):
            raise ValueError(
                "Silent-frame manifold filter removed too many train states: "
                f"kept {int(manifold_mask.sum())} of {manifold_mask.size}"
            )
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

    landmark_count = int(cfg.get("landmark_count", min(750, f_train_flat.shape[0])))
    manifold_indices = np.flatnonzero(manifold_mask)
    f_manifold = f_train_flat[manifold_mask]
    x_manifold = x_train_flat[manifold_mask]
    delta_manifold = delta_train_flat[manifold_mask]
    raw_delta_manifold = raw_delta_train_flat[manifold_mask]
    lm_idx, landmark_meta = select_landmarks(
        f_manifold,
        x_manifold,
        raw_delta_manifold,
        landmark_count,
        rng,
        cfg,
    )
    lm_idx = manifold_indices[lm_idx]
    f_lm = f_train_flat[lm_idx]
    df_lm = delta_train_flat[lm_idx]
    print(
        f"Selected train landmarks: {f_lm.shape[0]} of {f_train_flat.shape[0]} states "
        f"({landmark_meta['mode']}: activity={landmark_meta['activity_count']}, "
        f"transition={landmark_meta['transition_count']}, coverage={landmark_meta['coverage_count']})"
    )

    d_mind, dist_meta = transition_geodesic_distances(
        f_lm,
        df_lm,
        k=int(cfg.get("transition_k", 16)),
        transition_weight=float(cfg.get("transition_weight", 1.0)),
        temperature=float(cfg.get("transition_temperature", 1.0)),
        sym=str(cfg.get("graph_sym", "min")),
        probability_mode=str(cfg.get("transition_probability_mode", "next_state")),
        exclude_self=bool(cfg.get("transition_exclude_self", True)),
        use_graph_geodesics=bool(cfg.get("use_graph_geodesics", True)),
    )
    print(f"Built transition/geodesic distances: max={dist_meta['distance_max']:.4f}, disconnected_replaced={dist_meta['disconnected_pairs_replaced']}")

    latent_dim = int(cfg.get("latent_dim", cfg.get("embedding_dim", 7)))
    z_lm, embed_meta = embed_distances(d_mind, latent_dim, cfg)
    print(f"Embedded landmarks: Z={z_lm.shape}, mode={embed_meta['embedding_mode']}")

    lle_k = int(cfg.get("lle_k", cfg.get("lle_neighbors", 16)))
    lle_ridge = float(cfg.get("lle_ridge", 1e-3))
    f2z = LLEMapper(lle_k, lle_ridge).fit(f_lm, z_lm)

    print("Mapping train and held-out states into the landmark manifold with LLE...")
    z_train = f2z.transform(f_train_flat)
    z_test = f2z.transform(f_test_flat)

    # Landmarks define the MIND/Sammon geometry, but reconstruction should not be
    # bottlenecked through landmarks. Fit the inverse decoder on every mapped
    # training state so held-out reconstruction uses the full training manifold.
    inverse_decoder = str(cfg.get("inverse_decoder", "kernel")).lower()
    inverse_k = int(cfg.get("inverse_k", cfg.get("kernel_k", lle_k)))
    kernel_bandwidth = float(cfg.get("kernel_bandwidth", 1.0))
    decoder_target = str(cfg.get("decoder_target", "raw")).lower()
    if decoder_target in {"raw", "neural", "activity"}:
        decoder_train_target = x_train_flat
    elif decoder_target in {"feature", "pca", "geometry"}:
        decoder_train_target = f_train_flat
    else:
        raise ValueError(f"Unknown decoder_target={decoder_target!r}")
    if inverse_decoder in {"kernel", "rbf", "weighted", "weighted_average"}:
        z2f = KernelRegressor(inverse_k, kernel_bandwidth).fit(z_train, decoder_train_target)
        print(
            f"Fitted inverse kernel decoder on all train states: {z_train.shape[0]} states, "
            f"k={inverse_k}, bandwidth={kernel_bandwidth:g}, target={decoder_target}"
        )
    elif inverse_decoder in {"lle", "local_linear"}:
        z2f = LLEMapper(inverse_k, lle_ridge).fit(z_train, decoder_train_target)
        print(f"Fitted inverse LLE decoder on all train states: {z_train.shape[0]} states, k={inverse_k}, target={decoder_target}")
    else:
        raise ValueError(f"Unknown inverse_decoder={inverse_decoder!r}")
    decoded_train = z2f.transform(z_train)
    decoded_test = z2f.transform(z_test)

    def inverse_features(fhat: np.ndarray) -> np.ndarray:
        xhat = pca.inverse_transform(fhat).astype(np.float32) if pca is not None else fhat.astype(np.float32)
        if norm_mu is not None and norm_sd is not None:
            xhat = (xhat * norm_sd + norm_mu).astype(np.float32)
        return xhat

    if decoder_target in {"raw", "neural", "activity"}:
        xhat_train_flat = decoded_train.astype(np.float32)
        xhat_test_flat = decoded_test.astype(np.float32)
    else:
        xhat_train_flat = inverse_features(decoded_train)
        xhat_test_flat = inverse_features(decoded_test)
    xhat_train_flat, xhat_test_flat, postprocess_meta = apply_reconstruction_postprocess(
        xhat_train_flat,
        xhat_test_flat,
        x_train_flat,
        cfg,
    )
    if postprocess_meta.get("enabled"):
        print(
            "Reconstruction postprocess: "
            f"floor={postprocess_meta.get('recon_activity_floor', 0.0):.4g}, "
            f"train R2 {postprocess_meta.get('train_R2_before_postprocess', float('nan')):.4f}"
            f" -> {postprocess_meta.get('train_R2_after_postprocess', float('nan')):.4f}"
        )
    xhat_train = xhat_train_flat.reshape(x_train_proc.shape)
    xhat_test = xhat_test_flat.reshape(x_test_proc.shape)

    train_metrics = corr_and_r2(x_train_proc, xhat_train)
    test_metrics = corr_and_r2(x_test_proc, xhat_test)
    event_metrics = compute_event_metrics(x_test_proc, xhat_test, percentile=float(cfg.get("event_metric_percentile", 99.0)))
    print(f"Train r {train_metrics['r']:.4f} | R2 {train_metrics['R2']:.4f}")
    print(f"Heldout r {test_metrics['r']:.4f} | R2 {test_metrics['R2']:.4f}")
    print(f"Event capture top1 {event_metrics.get('top_1_percent_event_capture', float('nan')):.4f} | dyn ratio {event_metrics.get('pred_dynamics_std_over_true_dynamics_std', float('nan')):.4f}")

    np.savez_compressed(
        out_dir / "analysis_cache_best.npz",
        y_valid=x_test_proc.astype(np.float32),
        xhat_valid=xhat_test.astype(np.float32),
        z_valid=z_test.reshape(x_test.shape[0], x_test.shape[1], -1).astype(np.float32),
        y_train=x_train_proc.astype(np.float32),
        xhat_train=xhat_train.astype(np.float32),
        z_train=z_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype(np.float32),
        z_landmarks=z_lm.astype(np.float32),
        landmark_indices=lm_idx.astype(np.int64),
        mind_distances=d_mind.astype(np.float32),
        train_trial_ids=trial_ids[train_idx],
        test_trial_ids=trial_ids[test_idx],
        neuron_mask=mask,
        axis=axis,
        metrics_json=json.dumps({"train": train_metrics, "heldout": test_metrics, "events": event_metrics}),
        postprocess_json=json.dumps(postprocess_meta),
        landmark_selection_json=json.dumps(landmark_meta),
        manifold_filter_json=json.dumps(manifold_meta),
    )
    save_recon_heatmap(x_test_proc, xhat_test, out_dir / "raw_vs_recon_t0_15_n0_40.png", "MIND-LLE held-out reconstruction")
    save_embedding_plot(z_lm, out_dir / "latent_manifold_mds.png")

    final_metrics = {
        "train": train_metrics,
        "heldout": test_metrics,
        "events": event_metrics,
        "R2": test_metrics["R2"],
        "r": test_metrics["r"],
    }
    torch.save(final_metrics, out_dir / "final_metrics.pt")

    metadata = {
        "script": Path(__file__).name,
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(data_path),
        "n_trials_total": int(sequences.shape[0]),
        "n_trials_train": int(len(train_idx)),
        "n_trials_heldout": int(len(test_idx)),
        "heldout_fraction": float(len(test_idx) / sequences.shape[0]),
        "sequence_length": int(sequences.shape[1]),
        "n_neurons": int(x_train.shape[-1]),
        "globally_silent_neurons_removed": int(removed),
        "feature_dim": int(f_train_flat.shape[1]),
        "use_pca": bool(use_pca),
        "pca_explained_variance": pca_var,
        "landmark_count": int(f_lm.shape[0]),
        "landmark_selection": landmark_meta,
        "manifold_filter": manifold_meta,
        "latent_dim": int(latent_dim),
        "split_seed": int(cfg.get("mind_split_seed", seed)),
        "train_trial_ids": [int(x) if float(x).is_integer() else float(x) for x in np.asarray(trial_ids[train_idx], dtype=float)],
        "heldout_trial_ids": [int(x) if float(x).is_integer() else float(x) for x in np.asarray(trial_ids[test_idx], dtype=float)],
        "transition_distance_meta": dist_meta,
        "embedding_meta": embed_meta,
        "lle_k": int(lle_k),
        "lle_ridge": float(lle_ridge),
        "inverse_decoder": inverse_decoder,
        "inverse_k": int(inverse_k),
        "kernel_bandwidth": float(kernel_bandwidth),
        "decoder_target": decoder_target,
        "inverse_fit_states": int(z_train.shape[0]),
        "postprocess_meta": postprocess_meta,
        "metrics": final_metrics,
        "method_note": "Train-only MIND-style local population geometry: PCA features, event-aware landmarks, graph geodesic distances, Sammon/MDS embedding, LLE f->z mapping from landmarks, and inverse z->PCA-feature decoder fit on all mapped train states for held-out trial reconstruction.",
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "training_results.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(final_metrics, indent=2))
        f.write("\n")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
