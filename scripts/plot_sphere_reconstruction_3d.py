#!/usr/bin/env python3
"""Plot true sphere trajectories and decoded reconstruction paths in 3D.

This is for the simulated sphere/great-circle experiments. It uses the known
place-field centers to decode predicted neural activity back onto the unit
sphere with a population-vector decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


TRUE_COLOR = "#12355B"
RECON_COLOR = "#FF5A1F"
SPHERE_COLOR = "#8E8E8E"
WIREFRAME_COLOR = "#5F5F5F"
TRUE_LINEWIDTH = 3.4
RECON_LINEWIDTH = 1.6


def first_key(data: np.lib.npyio.NpzFile, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of {list(names)} found in {data.filename}; keys={data.files}")


def infer_source_from_run(run_dir: Path) -> Path | None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    data_path_raw = metadata.get("data_path")
    if not data_path_raw:
        return None
    data_path = Path(data_path_raw)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        return None
    data = np.load(data_path, allow_pickle=True)
    if "source_file" not in data.files:
        return None
    source = Path(str(np.asarray(data["source_file"]).item()))
    if not source.is_absolute():
        cwd_source = Path.cwd() / source
        source = cwd_source if cwd_source.exists() else data_path.parent / source
    return source if source.exists() else None


def theta_phi_to_xyz(theta_phi: np.ndarray) -> np.ndarray:
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


def centers_to_xyz(theta_centers: np.ndarray, phi_centers: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.sin(theta_centers) * np.cos(phi_centers),
            np.sin(theta_centers) * np.sin(phi_centers),
            np.cos(theta_centers),
        ],
        axis=1,
    )


def population_vector_decode(rates: np.ndarray, centers_xyz: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(rates, dtype=np.float64), 0.0, None)
    xyz = weights @ centers_xyz
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(norm, 1e-12)


def angular_error_deg(true_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    dot = np.sum(true_xyz * pred_xyz, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def load_source(source_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.load(source_path, allow_pickle=True)
    true_latents = first_key(source, ["true_latents", "latents_theta_phi"]).astype(np.float64)
    theta_centers = source["theta_centers"].astype(np.float64)
    phi_centers = source["phi_centers"].astype(np.float64)
    return true_latents, theta_centers, phi_centers


def fit_output_series(run_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    fit_path = run_dir / "fit_outputs.npz"
    fit = np.load(fit_path, allow_pickle=True)
    true_rates = first_key(fit, ["activities", "true_rates"])
    true_latents = first_key(fit, ["true_latents", "latents_theta_phi"])
    series = []
    prediction_keys = [
        ("geodesic", ["rates_geo_pred", "pred_geo"]),
        ("free", ["rates_free_pred", "pred_free"]),
    ]
    for label, keys in prediction_keys:
        for key in keys:
            if key in fit.files:
                series.append((label, true_rates, fit[key], true_latents))
                break
    return series


def analysis_cache_series(
    run_dir: Path,
    true_latents_source: np.ndarray,
    split: str,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    cache = np.load(run_dir / "analysis_cache_best.npz", allow_pickle=True)
    if split == "all" and {"x_train_true", "x_train_pred", "train_frame_trial_ids"}.issubset(cache.files):
        true_rates = np.concatenate(
            [
                cache["x_train_true"].reshape(-1, cache["x_train_true"].shape[-1]),
                cache["x_true"].reshape(-1, cache["x_true"].shape[-1]),
            ],
            axis=0,
        )
        pred_rates = np.concatenate(
            [
                cache["x_train_pred"].reshape(-1, cache["x_train_pred"].shape[-1]),
                cache["x_pred"].reshape(-1, cache["x_pred"].shape[-1]),
            ],
            axis=0,
        )
        frame_trial_ids = np.concatenate(
            [cache["train_frame_trial_ids"].astype(int), cache["test_frame_trial_ids"].astype(int)],
            axis=0,
        )
    else:
        true_rates = cache["x_true"].reshape(-1, cache["x_true"].shape[-1])
        pred_rates = cache["x_pred"].reshape(-1, cache["x_pred"].shape[-1])
        frame_trial_ids = cache["test_frame_trial_ids"].astype(int) if "test_frame_trial_ids" in cache.files else None
    return [("v6_geometric_ae", true_rates, pred_rates, true_latents_source, frame_trial_ids)]


def draw_plot(
    *,
    true_latents: np.ndarray,
    pred_rates: np.ndarray,
    centers_xyz: np.ndarray,
    out_path: Path,
    label: str,
    true_rates: np.ndarray | None = None,
    frame_trial_ids: np.ndarray | None = None,
    max_trials: int | None = None,
) -> None:
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 96)
    v = np.linspace(0, np.pi, 48)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=SPHERE_COLOR, alpha=0.26, linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color=WIREFRAME_COLOR, alpha=0.12, linewidth=0.45, rstride=5, cstride=5)

    errors: list[np.ndarray] = []
    if frame_trial_ids is None:
        n_trials = pred_rates.shape[0]
        trial_ids = np.arange(n_trials)
        if max_trials is not None:
            trial_ids = trial_ids[:max_trials]
        for trial_id in trial_ids:
            pred = pred_rates[trial_id]
            n = min(pred.shape[0], true_latents[trial_id].shape[0])
            true_xyz = theta_phi_to_xyz(true_latents[trial_id, :n])
            pred_xyz = population_vector_decode(pred[:n], centers_xyz)
            errors.append(angular_error_deg(true_xyz, pred_xyz))
            step = max(1, n // 260)
            ax.plot(true_xyz[::step, 0], true_xyz[::step, 1], true_xyz[::step, 2], color=TRUE_COLOR, lw=TRUE_LINEWIDTH, alpha=0.88)
            ax.plot(pred_xyz[::step, 0], pred_xyz[::step, 1], pred_xyz[::step, 2], color=RECON_COLOR, lw=RECON_LINEWIDTH, alpha=0.98)
    else:
        trial_ids = np.unique(frame_trial_ids)
        if max_trials is not None:
            trial_ids = trial_ids[:max_trials]
        for trial_id in trial_ids:
            mask = frame_trial_ids == trial_id
            pred = pred_rates[mask]
            n = min(pred.shape[0], true_latents[trial_id].shape[0])
            true_xyz = theta_phi_to_xyz(true_latents[trial_id, :n])
            pred_xyz = population_vector_decode(pred[:n], centers_xyz)
            errors.append(angular_error_deg(true_xyz, pred_xyz))
            step = max(1, n // 260)
            ax.plot(true_xyz[::step, 0], true_xyz[::step, 1], true_xyz[::step, 2], color=TRUE_COLOR, lw=TRUE_LINEWIDTH, alpha=0.88)
            ax.plot(pred_xyz[::step, 0], pred_xyz[::step, 1], pred_xyz[::step, 2], color=RECON_COLOR, lw=RECON_LINEWIDTH, alpha=0.98)

    err = np.concatenate(errors) if errors else np.array([np.nan])
    display_label = label.replace("_", " ")
    title = f"True vs Predicted Sphere Trajectories ({display_label})"
    ax.set_title(title, pad=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=24, azim=38)
    ax.legend(
        handles=[
            Line2D([0], [0], color=TRUE_COLOR, lw=TRUE_LINEWIDTH, label="true path"),
            Line2D([0], [0], color=RECON_COLOR, lw=RECON_LINEWIDTH, label="predicted path"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"{out_path} | mean={np.nanmean(err):.6f} median={np.nanmedian(err):.6f} max={np.nanmax(err):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", type=Path, default=None, help="synthetic_sphere_trials.npz source file.")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--split", choices=["heldout", "all"], default="heldout")
    args = parser.parse_args()

    run_dir = args.run_dir
    source_path = args.source or infer_source_from_run(run_dir)
    source_latents = None
    theta_centers = None
    phi_centers = None
    if source_path is not None:
        source_latents, theta_centers, phi_centers = load_source(source_path)

    if (run_dir / "fit_outputs.npz").exists():
        for label, true_rates, pred_rates, fit_latents in fit_output_series(run_dir):
            if theta_centers is None or phi_centers is None:
                fallback_source = run_dir / "synthetic_sphere_trials.npz"
                if not fallback_source.exists():
                    raise FileNotFoundError(f"Need --source or {fallback_source}")
                source_latents, theta_centers, phi_centers = load_source(fallback_source)
            centers_xyz = centers_to_xyz(theta_centers, phi_centers)
            out_path = run_dir / f"sphere_reconstruction_3d_overlay_{label}.png"
            draw_plot(
                true_latents=fit_latents,
                pred_rates=pred_rates,
                true_rates=true_rates,
                centers_xyz=centers_xyz,
                out_path=out_path,
                label=label,
                max_trials=args.max_trials,
            )
        return

    if (run_dir / "analysis_cache_best.npz").exists():
        if source_latents is None or theta_centers is None or phi_centers is None:
            raise FileNotFoundError(f"Could not infer source sphere NPZ for {run_dir}; pass --source.")
        centers_xyz = centers_to_xyz(theta_centers, phi_centers)
        for label, true_rates, pred_rates, true_latents, frame_trial_ids in analysis_cache_series(run_dir, source_latents, args.split):
            suffix = "" if args.split == "heldout" else f"_{args.split}"
            out_path = run_dir / f"sphere_reconstruction_3d_overlay{suffix}.png"
            draw_plot(
                true_latents=true_latents,
                pred_rates=pred_rates,
                true_rates=true_rates,
                centers_xyz=centers_xyz,
                out_path=out_path,
                label=label,
                frame_trial_ids=frame_trial_ids,
                max_trials=args.max_trials,
            )
        return

    raise FileNotFoundError(f"No fit_outputs.npz or analysis_cache_best.npz in {run_dir}")


if __name__ == "__main__":
    main()
