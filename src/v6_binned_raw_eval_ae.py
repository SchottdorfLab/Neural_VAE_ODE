#!/usr/bin/env python3
"""Position-binned geometric AE with mapped-back raw-frame evaluation.

This experiment trains the current v6 geometric AE on fixed-length binned
trials, then interpolates each heldout trial's binned reconstruction back onto
the original raw frame positions. The goal is to test whether binning can make
the model easier to train while still exposing reconstruction quality on the
real, unbinned neural frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

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


def normalized_position_range(
    position: np.ndarray,
    trials: np.ndarray,
    trial_ids: np.ndarray,
    cfg: Dict[str, Any],
) -> tuple[float, float]:
    configured_lo = float(cfg.get("position_min", np.nan))
    configured_hi = float(cfg.get("position_max", np.nan))
    if np.isfinite(configured_lo) and np.isfinite(configured_hi) and configured_hi > configured_lo:
        return configured_lo, configured_hi

    all_pos = []
    for tid in np.asarray(trial_ids).reshape(-1):
        idx = np.flatnonzero(trials == int(tid))
        if idx.size == 0:
            continue
        pos = np.asarray(position[idx], dtype=np.float64)
        if cfg_bool(cfg, "position_cumulative_max", True):
            pos = np.maximum.accumulate(pos)
        all_pos.append(pos)
    if not all_pos:
        raise ValueError("No positions found for selected trials.")
    merged = np.concatenate(all_pos)
    return float(np.nanmin(merged)), float(np.nanmax(merged))


def raw_axis_for_trial(
    idx: np.ndarray,
    time: np.ndarray | None,
    position: np.ndarray | None,
    trials: np.ndarray,
    trial_ids_for_range: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    align = str(cfg.get("sequence_alignment", cfg.get("time_mode", "time"))).lower()
    if align in {"position", "pos", "maze_position"}:
        if position is None:
            raise KeyError("sequence_alignment=position requires Position in the NPZ")
        axis = np.asarray(position[idx], dtype=np.float64)
        if cfg_bool(cfg, "position_cumulative_max", True):
            axis = np.maximum.accumulate(axis)
        if cfg_bool(cfg, "position_normalize", True):
            lo, hi = normalized_position_range(position, trials, trial_ids_for_range, cfg)
            if hi > lo:
                axis = np.clip((axis - lo) / (hi - lo), 0.0, 1.0)
        return axis.astype(np.float32)

    if time is not None:
        axis = np.asarray(time[idx], dtype=np.float64)
        axis = axis - axis[0]
    else:
        axis = np.arange(idx.size, dtype=np.float64) / cfg_float(cfg, "fps", 10.0)
    if cfg_bool(cfg, "normalize_time", True):
        lo, hi = float(axis[0]), float(axis[-1])
        axis = (axis - lo) / (hi - lo) if hi > lo else np.zeros_like(axis)
    return axis.astype(np.float32)


def build_raw_frame_split_for_binned_trials(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    roi, trials, time, position = mind.load_neural_data(data["data_path"])
    roi = roi[:, data["mask"]].astype(np.float32)
    trials = trials.astype(np.int64)
    trial_ids = np.asarray(data["trial_ids"], dtype=np.int64)
    train_trial_ids = trial_ids[data["train_idx"]]
    test_trial_ids = trial_ids[data["test_idx"]]

    keep_trials = np.isin(trials, trial_ids)
    if cfg_bool(cfg, "raw_eval_filter_silent_frames", True):
        keep_trials &= np.sum(roi, axis=1) > cfg_float(cfg, "silent_frame_threshold", 0.0)

    raw_data = roi[keep_trials].astype(np.float32)
    raw_trial_ids = trials[keep_trials].astype(np.int64)
    raw_source_idx = np.flatnonzero(keep_trials).astype(np.int64)
    raw_time = np.asarray(time[keep_trials], dtype=np.float32) if time is not None else raw_source_idx.astype(np.float32)
    if position is not None:
        raw_position = np.asarray(position[keep_trials], dtype=np.float32)
    else:
        raw_position = np.full(raw_data.shape[0], np.nan, dtype=np.float32)

    raw_axis = np.empty(raw_data.shape[0], dtype=np.float32)
    for tid in trial_ids:
        local = np.flatnonzero(raw_trial_ids == int(tid))
        if local.size == 0:
            continue
        original_idx = raw_source_idx[local]
        raw_axis[local] = raw_axis_for_trial(original_idx, time, position, trials, trial_ids, cfg)

    train_mask = np.isin(raw_trial_ids, train_trial_ids)
    test_mask = np.isin(raw_trial_ids, test_trial_ids)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Raw mapped split has empty train or heldout frames.")

    return {
        "train_x": raw_data[train_mask][None, :, :],
        "test_x": raw_data[test_mask][None, :, :],
        "train_frame_trial_ids": raw_trial_ids[train_mask],
        "test_frame_trial_ids": raw_trial_ids[test_mask],
        "train_frame_axis": raw_axis[train_mask],
        "test_frame_axis": raw_axis[test_mask],
        "train_frame_time": raw_time[train_mask],
        "test_frame_time": raw_time[test_mask],
        "train_frame_position": raw_position[train_mask],
        "test_frame_position": raw_position[test_mask],
        "train_source_frame_index": raw_source_idx[train_mask],
        "test_source_frame_index": raw_source_idx[test_mask],
        "train_trial_ids": train_trial_ids,
        "test_trial_ids": test_trial_ids,
        "n_train_frames": int(train_mask.sum()),
        "n_test_frames": int(test_mask.sum()),
        "axis_kind": str(cfg.get("sequence_alignment", "position")),
        "filtered_silent_frames": cfg_bool(cfg, "raw_eval_filter_silent_frames", True),
    }


def map_binned_predictions_to_raw_frames(
    binned_pred: np.ndarray,
    binned_trial_ids: np.ndarray,
    raw_frame_trial_ids: np.ndarray,
    raw_frame_axis: np.ndarray,
    bin_axis: np.ndarray,
) -> np.ndarray:
    binned_pred = np.asarray(binned_pred, dtype=np.float32)
    out = np.empty((raw_frame_trial_ids.size, binned_pred.shape[-1]), dtype=np.float32)
    trial_to_row = {int(tid): i for i, tid in enumerate(np.asarray(binned_trial_ids).reshape(-1))}
    for tid in np.unique(raw_frame_trial_ids):
        raw_sel = raw_frame_trial_ids == int(tid)
        row = trial_to_row.get(int(tid))
        if row is None:
            raise KeyError(f"Raw trial {tid} is missing from binned predictions.")
        seq = binned_pred[row]
        axis = np.clip(raw_frame_axis[raw_sel].astype(np.float64), float(bin_axis[0]), float(bin_axis[-1]))
        for n in range(seq.shape[1]):
            out[raw_sel, n] = np.interp(axis, bin_axis, seq[:, n]).astype(np.float32)
    return out


def save_raw_mapped_heatmap(
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
    import matplotlib.pyplot as plt

    ids = np.asarray(frame_trial_ids).reshape(-1)
    times = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    unique_ids = np.unique(ids)
    pick = int(np.clip(trial_index, 0, unique_ids.size - 1))
    trial_id = unique_ids[pick]
    frame_idx = np.flatnonzero(ids == int(trial_id))
    true_trial = y_true[0, frame_idx, :]
    pred_trial = y_pred[0, frame_idx, :]
    time_trial = times[frame_idx] - times[frame_idx][0]

    f0 = max(0, int(frame_start))
    f1 = min(int(frame_end) + 1, true_trial.shape[0])
    n0 = max(0, int(neuron_start))
    n1 = min(int(neuron_end) + 1, true_trial.shape[1])
    if f0 >= f1 or n0 >= n1:
        raise ValueError(f"Requested empty raw mapped window; trial shape={true_trial.shape}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v6_binned_raw_eval_ae.txt")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config key=value")
    args = parser.parse_args()

    cfg = mind.load_config(args.config)
    apply_overrides(cfg, args.overrides)
    seed = cfg_int(cfg, "seed", cfg_int(cfg, "mind_split_seed", 42))
    neural.set_torch_seed(seed)
    device = neural.device_from_config(cfg)
    print(f"Device: {device}")

    out_dir = Path(cfg.get("out_dir", "runs/v6_binned_raw_eval_ae_manual")).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = neural.prepare_data(cfg, seed)
    teacher = neural.build_mind_teacher(cfg, data)
    model, train_meta = neural.train_model(cfg, data, teacher, device)

    fhat_train, zhat_train = neural.predict_features(model, data["f_train_flat"], device, cfg_int(cfg, "predict_batch_size", 4096))
    fhat_test, zhat_test = neural.predict_features(model, data["f_test_flat"], device, cfg_int(cfg, "predict_batch_size", 4096))

    prediction_decoder = str(cfg.get("prediction_decoder", "neural")).lower()
    prediction_meta: Dict[str, Any] = {"decoder": prediction_decoder}
    if prediction_decoder in {"local_kernel", "kernel", "mind_local", "local"}:
        target = str(cfg.get("local_prediction_target", cfg.get("decoder_target", "feature"))).lower()
        k = cfg_int(cfg, "local_prediction_k", cfg_int(cfg, "inverse_k", 128))
        bandwidth = cfg_float(cfg, "local_prediction_bandwidth", cfg_float(cfg, "kernel_bandwidth", 0.35))
        print(
            "Using local kernel prediction decoder instead of neural decoder: "
            f"z -> {target}, k={k}, bandwidth={bandwidth:g}"
        )
        if target in {"raw", "neural", "activity"}:
            local_target = data["x_train_flat"]
        elif target in {"feature", "pca", "geometry"}:
            local_target = data["f_train_flat"]
        else:
            raise ValueError(f"Unknown local_prediction_target={target!r}")
        local_decoder = mind.KernelRegressor(k, bandwidth).fit(zhat_train, local_target)
        local_train = local_decoder.transform(zhat_train).astype(np.float32)
        local_test = local_decoder.transform(zhat_test).astype(np.float32)
        if target in {"raw", "neural", "activity"}:
            xhat_train_flat = local_train
            xhat_test_flat = local_test
            fhat_train = data["f_train_flat"]
            fhat_test = data["f_test_flat"]
        else:
            fhat_train = local_train
            fhat_test = local_test
            xhat_train_flat = neural.inverse_features_numpy(fhat_train, data["pca"], data["norm_mu"], data["norm_sd"])
            xhat_test_flat = neural.inverse_features_numpy(fhat_test, data["pca"], data["norm_mu"], data["norm_sd"])
        prediction_meta.update({"target": target, "k": int(k), "bandwidth": float(bandwidth)})
    elif prediction_decoder in {"local_lle", "lle", "local_linear"}:
        target = str(cfg.get("local_prediction_target", cfg.get("decoder_target", "feature"))).lower()
        k = cfg_int(cfg, "local_prediction_k", cfg_int(cfg, "inverse_k", 128))
        ridge = cfg_float(cfg, "local_prediction_ridge", cfg_float(cfg, "lle_ridge", 0.001))
        print(
            "Using local LLE prediction decoder instead of neural decoder: "
            f"z -> {target}, k={k}, ridge={ridge:g}"
        )
        if target in {"raw", "neural", "activity"}:
            local_target = data["x_train_flat"]
        elif target in {"feature", "pca", "geometry"}:
            local_target = data["f_train_flat"]
        else:
            raise ValueError(f"Unknown local_prediction_target={target!r}")
        local_decoder = mind.LLEMapper(k, ridge).fit(zhat_train, local_target)
        local_train = local_decoder.transform(zhat_train).astype(np.float32)
        local_test = local_decoder.transform(zhat_test).astype(np.float32)
        if target in {"raw", "neural", "activity"}:
            xhat_train_flat = local_train
            xhat_test_flat = local_test
            fhat_train = data["f_train_flat"]
            fhat_test = data["f_test_flat"]
        else:
            fhat_train = local_train
            fhat_test = local_test
            xhat_train_flat = neural.inverse_features_numpy(fhat_train, data["pca"], data["norm_mu"], data["norm_sd"])
            xhat_test_flat = neural.inverse_features_numpy(fhat_test, data["pca"], data["norm_mu"], data["norm_sd"])
        prediction_meta.update({"target": target, "k": int(k), "ridge": float(ridge)})
    elif prediction_decoder in {"neural", "mlp", "global", "global_neural"}:
        xhat_train_flat = neural.inverse_features_numpy(fhat_train, data["pca"], data["norm_mu"], data["norm_sd"])
        xhat_test_flat = neural.inverse_features_numpy(fhat_test, data["pca"], data["norm_mu"], data["norm_sd"])
    else:
        raise ValueError(f"Unknown prediction_decoder={prediction_decoder!r}")
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

    binned_train_metrics = mind.corr_and_r2(data["x_train_proc"], xhat_train)
    binned_train_metrics.update(mind.summarize_trial_mind_r2(data["x_train_proc"], xhat_train))
    binned_test_metrics = mind.corr_and_r2(data["x_test_proc"], xhat_test)
    binned_test_metrics.update(mind.summarize_trial_mind_r2(data["x_test_proc"], xhat_test))
    binned_event_metrics = mind.compute_event_metrics(data["x_test_proc"], xhat_test, percentile=cfg_float(cfg, "event_metric_percentile", 99.0))

    raw_eval = build_raw_frame_split_for_binned_trials(cfg, data)
    train_trial_ids = data["trial_ids"][data["train_idx"]]
    test_trial_ids = data["trial_ids"][data["test_idx"]]
    bin_axis = np.asarray(data["axis"], dtype=np.float64)
    xhat_train_raw_flat = map_binned_predictions_to_raw_frames(
        xhat_train,
        train_trial_ids,
        raw_eval["train_frame_trial_ids"],
        raw_eval["train_frame_axis"],
        bin_axis,
    )
    xhat_test_raw_flat = map_binned_predictions_to_raw_frames(
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
        raw_eval["train_x"], xhat_train_raw, raw_eval["train_frame_trial_ids"], raw_eval["train_trial_ids"]
    ))
    raw_test_metrics = mind.corr_and_r2(raw_eval["test_x"], xhat_test_raw)
    raw_test_metrics.update(mind.summarize_frame_trial_mind_r2(
        raw_eval["test_x"], xhat_test_raw, raw_eval["test_frame_trial_ids"], raw_eval["test_trial_ids"]
    ))
    raw_event_metrics = mind.compute_event_metrics(raw_eval["test_x"], xhat_test_raw, percentile=cfg_float(cfg, "event_metric_percentile", 99.0))

    print(
        f"Binned heldout r {binned_test_metrics['r']:.4f} | R2 {binned_test_metrics['R2']:.4f} | "
        f"MIND_R2 {binned_test_metrics['MIND_R2']:.4f} | trial median {binned_test_metrics['MIND_R2_trial_median']:.4f}"
    )
    print(
        f"Raw mapped heldout r {raw_test_metrics['r']:.4f} | R2 {raw_test_metrics['R2']:.4f} | "
        f"MIND_R2 {raw_test_metrics['MIND_R2']:.4f} | trial median {raw_test_metrics['MIND_R2_trial_median']:.4f}"
    )
    print(
        f"Raw mapped event capture top1 {raw_event_metrics['top_1_percent_event_capture']:.4f} | "
        f"dyn ratio {raw_event_metrics['pred_dynamics_std_over_true_dynamics_std']:.4f}"
    )

    torch.save({"model_state": model.state_dict(), "config": jsonable(cfg), "train_meta": jsonable(train_meta)}, out_dir / "model.pt")
    np.savez_compressed(
        out_dir / "analysis_cache_best.npz",
        x_true=raw_eval["test_x"].astype(np.float32),
        x_pred=xhat_test_raw.astype(np.float32),
        x_binned_true=data["x_test_proc"].astype(np.float32),
        x_binned_pred=xhat_test.astype(np.float32),
        x_train_true=raw_eval["train_x"].astype(np.float32),
        x_train_pred=xhat_train_raw.astype(np.float32),
        z_test=zhat_test.reshape(x_test_shape[0], x_test_shape[1], -1).astype(np.float32),
        z_train=zhat_train.reshape(x_train_shape[0], x_train_shape[1], -1).astype(np.float32),
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
        metrics_json=np.asarray(json.dumps(jsonable({"binned": binned_test_metrics, "raw_mapped": raw_test_metrics}))),
    )
    save_raw_mapped_heatmap(
        raw_eval["test_x"],
        xhat_test_raw,
        out_dir / "raw_vs_recon_t0_15_n0_40.png",
        "Binned-trained reconstruction mapped to raw frames",
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
        "Binned heldout reconstruction",
    )
    mind.save_embedding_plot(teacher["z_lm"], out_dir / "latent_manifold_mds.png")

    final_metrics = {
        "binned_train": binned_train_metrics,
        "binned_heldout": binned_test_metrics,
        "binned_events": binned_event_metrics,
        "raw_mapped_train": raw_train_metrics,
        "raw_mapped_heldout": raw_test_metrics,
        "raw_mapped_events": raw_event_metrics,
        "R2": raw_test_metrics["R2"],
        "r": raw_test_metrics["r"],
    }
    torch.save(final_metrics, out_dir / "final_metrics.pt")
    metadata = {
        "script": Path(__file__).name,
        "config_path": str(Path(args.config).resolve()),
        "data_path": str(data["data_path"]),
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
        "postprocess_meta": postprocess_meta,
        "prediction_decoder": prediction_meta,
        "raw_eval": {k: v for k, v in raw_eval.items() if not isinstance(v, np.ndarray)},
        "transition_distance_meta": teacher["dist_meta"],
        "embedding_meta": teacher["embed_meta"],
        "landmark_selection": teacher["landmark_meta"],
        "manifold_filter": teacher["manifold_meta"],
        "mapping_cv": {
            "f2z": teacher.get("f2z_cv_meta"),
            "inverse": teacher.get("inverse_cv_meta"),
            "selected_lle_k": teacher.get("selected_lle_k"),
            "selected_lle_ridge": teacher.get("selected_lle_ridge"),
            "selected_inverse_k": teacher.get("selected_inverse_k"),
            "selected_kernel_bandwidth": teacher.get("selected_kernel_bandwidth"),
        },
        "training": train_meta,
        "metrics": final_metrics,
        "method_note": (
            "Train the v6 geometric AE on fixed position-bin sequences, then interpolate "
            "the binned heldout reconstructions back to each original raw frame position "
            "for raw-frame reconstruction metrics and visualization."
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2) + "\n")
    (out_dir / "training_results.txt").write_text(json.dumps(jsonable(final_metrics), indent=2) + "\n")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
