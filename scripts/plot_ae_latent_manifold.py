#!/usr/bin/env python3
"""Plot train and heldout AE latent trajectories in a shared 3D PCA view."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA


def trajectory_collection(points: np.ndarray, cmap, linewidth: float, alpha: float) -> Line3DCollection:
    segments = np.stack([points[:-1], points[1:]], axis=1)
    progress = (np.arange(points.shape[0] - 1, dtype=np.float64) + 0.5) / (points.shape[0] - 1)
    collection = Line3DCollection(segments, cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    collection.set_array(progress)
    collection.set_linewidth(linewidth)
    collection.set_alpha(alpha)
    return collection


def equalize_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6) * 1.04
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache_path = args.run_dir / "analysis_cache_best.npz"
    cache = np.load(cache_path, allow_pickle=True)
    z_train = np.asarray(cache["z_train"], dtype=np.float64)
    z_test = np.asarray(cache["z_test"], dtype=np.float64)
    if z_train.ndim != 3 or z_test.ndim != 3:
        raise ValueError(f"Expected trial-shaped latent arrays, got {z_train.shape} and {z_test.shape}")
    if z_train.shape[2] < 3:
        raise ValueError(f"At least three latent dimensions are required, got {z_train.shape[2]}")

    pca = PCA(n_components=3)
    train_3d = pca.fit_transform(z_train.reshape(-1, z_train.shape[2])).reshape(z_train.shape[0], z_train.shape[1], 3)
    test_3d = pca.transform(z_test.reshape(-1, z_test.shape[2])).reshape(z_test.shape[0], z_test.shape[1], 3)
    all_3d = np.concatenate([train_3d.reshape(-1, 3), test_3d.reshape(-1, 3)], axis=0)

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")

    for trajectory in train_3d:
        ax.add_collection3d(trajectory_collection(trajectory, cmap, linewidth=0.55, alpha=0.18))
    for trajectory in test_3d:
        ax.add_collection3d(trajectory_collection(trajectory, cmap, linewidth=1.7, alpha=0.92))

    equalize_axes(ax, all_3d)
    explained = 100.0 * pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)", labelpad=10)
    ax.set_title(
        "E65 AE latent trajectories\n"
        f"7D latent space projected with train-only PCA; {z_train.shape[0]} train and {z_test.shape[0]} heldout trials",
        pad=18,
    )
    ax.view_init(elev=24, azim=-56)

    train_proxy = LineCollection([], linewidths=0.8, alpha=0.35, colors=[cmap(0.55)])
    test_proxy = LineCollection([], linewidths=2.2, alpha=1.0, colors=[cmap(0.55)])
    ax.legend([train_proxy, test_proxy], ["Train", "Heldout"], loc="upper left", frameon=False)

    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.68, pad=0.08)
    colorbar.set_label("Sequence progression")
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    output = args.output or (args.run_dir / "ae_latent_manifold_pca3d_time.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")


if __name__ == "__main__":
    main()
