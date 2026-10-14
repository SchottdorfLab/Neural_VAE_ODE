#!/usr/bin/env python3
"""Compare JAX and fit-sphere training on the same synthetic sphere data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = ROOT / "runs/geodesic_sphere_learned_metric_3d_6x80_e3000"
JAX_DIR = ROOT / "runs/jax_best_fit_sphere_data_6x80_e3000_lr1e4"
OUT_DIR = ROOT / "runs/jax_vs_fit_sphere_best_data_6x80_e3000"
OUT_PATH = OUT_DIR / "reconstruction_training_comparison.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(FIT_DIR / "fit_outputs.npz") as fit_npz:
        fit_objective = np.asarray(fit_npz["loss_free"], dtype=np.float64)
    with np.load(JAX_DIR / "fit_outputs.npz") as jax_npz:
        jax_objective = np.asarray(jax_npz["mse_history"], dtype=np.float64)

    fit_summary = load_json(FIT_DIR / "summary.json")
    jax_summary = load_json(JAX_DIR / "summary.json")
    fit_r2 = float(fit_summary["free"]["R2"])
    jax_r2 = float(jax_summary["jax_geodesic"]["train_R2"])

    fit_relative = 100.0 * fit_objective / fit_objective[0]
    jax_relative = 100.0 * jax_objective / jax_objective[0]
    fit_epochs = np.arange(fit_relative.size)
    jax_epochs = np.arange(jax_relative.size)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, (ax_curve, ax_r2) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        gridspec_kw={"width_ratios": [2.25, 1.0]},
    )

    jax_color = "#2563EB"
    fit_color = "#EA580C"

    ax_curve.plot(
        jax_epochs,
        jax_relative,
        color=jax_color,
        linewidth=2.0,
        label="JAX geodesic (MSE)",
    )
    ax_curve.plot(
        fit_epochs,
        fit_relative,
        color=fit_color,
        linewidth=2.0,
        label="Fit-sphere free dynamics (Poisson NLL)",
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Training objective (% of initial)")
    ax_curve.set_title("Optimization over time")
    ax_curve.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax_curve.legend(frameon=False, loc="upper right")
    ax_curve.set_xlim(0, max(fit_epochs[-1], jax_epochs[-1]))

    bars = ax_r2.bar(
        ["JAX\ngeodesic", "Fit-sphere\nfree"],
        [jax_r2, fit_r2],
        color=[jax_color, fit_color],
        width=0.62,
    )
    ax_r2.set_ylim(0.0, 1.05)
    ax_r2.set_ylabel("Train R²")
    ax_r2.set_title("Final reconstruction")
    ax_r2.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(bars, [jax_r2, fit_r2]):
        ax_r2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.suptitle("Same synthetic sphere data: 6 trials × 80 timepoints × 96 neurons", y=1.01)
    fig.text(
        0.5,
        -0.01,
        "Training-set comparison. Curves show relative progress because the models optimized different objectives.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
