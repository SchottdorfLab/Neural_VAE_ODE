#!/usr/bin/env python3
"""Analyze the learned 3D Riemannian metric from a sphere-trial run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class MetricNetwork(nn.Module):
    """Architecture used by fit_sphere_trials.py."""

    def __init__(self, latent_dim=3, hidden_dim=32):
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

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        L = torch.zeros(
            batch_size,
            self.latent_dim,
            self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                L[:, i, j] = torch.exp(out[:, idx]) if i == j else out[:, idx]
                idx += 1
        eye = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        return torch.bmm(L, L.transpose(1, 2)) + self.eps * eye


def metric_at(metric_net, z):
    return metric_net(z.unsqueeze(0))[0]


def christoffel_at(metric_net, z, create_graph=True):
    def metric_fn(point):
        return metric_at(metric_net, point)

    g = metric_fn(z)
    dg = torch.autograd.functional.jacobian(metric_fn, z, create_graph=create_graph)
    g_inv = torch.linalg.inv(g)
    d = z.numel()
    gamma = torch.zeros(d, d, d, dtype=z.dtype, device=z.device)
    for upper in range(d):
        for first in range(d):
            for second in range(d):
                value = 0.0
                for contracted in range(d):
                    value = value + 0.5 * g_inv[upper, contracted] * (
                        dg[contracted, second, first]
                        + dg[contracted, first, second]
                        - dg[first, second, contracted]
                    )
                gamma[upper, first, second] = value
    return g, gamma


def curvature_at(metric_net, point):
    z = point.detach().clone().requires_grad_(True)

    def gamma_fn(location):
        return christoffel_at(metric_net, location, create_graph=True)[1]

    g, gamma = christoffel_at(metric_net, z, create_graph=True)
    dgamma = torch.autograd.functional.jacobian(gamma_fn, z, create_graph=False)
    d = z.numel()
    riemann = torch.zeros(d, d, d, d, dtype=z.dtype, device=z.device)

    # R^rho_{ sigma mu nu } for R(X, Y)Z.
    for rho in range(d):
        for sigma in range(d):
            for mu in range(d):
                for nu in range(d):
                    value = dgamma[rho, nu, sigma, mu] - dgamma[rho, mu, sigma, nu]
                    for contracted in range(d):
                        value = value + (
                            gamma[rho, mu, contracted] * gamma[contracted, nu, sigma]
                            - gamma[rho, nu, contracted] * gamma[contracted, mu, sigma]
                        )
                    riemann[rho, sigma, mu, nu] = value

    ricci = torch.zeros(d, d, dtype=z.dtype, device=z.device)
    for sigma in range(d):
        for nu in range(d):
            ricci[sigma, nu] = sum(riemann[rho, sigma, rho, nu] for rho in range(d))

    g_inv = torch.linalg.inv(g)
    scalar = torch.sum(g_inv * ricci)

    # Columns of frame are orthonormal under g: frame.T @ g @ frame = I.
    chol = torch.linalg.cholesky(g)
    frame = torch.linalg.inv(chol).transpose(0, 1)
    ricci_orth = frame.transpose(0, 1) @ ricci @ frame
    ricci_orth = 0.5 * (ricci_orth + ricci_orth.transpose(0, 1))
    ricci_eigenvalues = torch.linalg.eigvalsh(ricci_orth)

    # In three dimensions, Ricci determines the Riemann tensor. These are the
    # eigenvalues of the curvature operator on the three principal 2-planes.
    sectional_eigenvalues = torch.sort(0.5 * scalar - ricci_eigenvalues).values

    frame_sectional = []
    for first in range(d):
        for second in range(first + 1, d):
            u = frame[:, first]
            v = frame[:, second]
            r_uv_v = torch.einsum("rsmn,s,m,n->r", riemann, v, u, v)
            frame_sectional.append(torch.dot(u, g @ r_uv_v))

    return {
        "metric": g.detach(),
        "gamma": gamma.detach(),
        "ricci": ricci.detach(),
        "scalar": scalar.detach(),
        "ricci_eigenvalues": ricci_eigenvalues.detach(),
        "sectional_eigenvalues": sectional_eigenvalues.detach(),
        "frame_sectional": torch.stack(frame_sectional).detach(),
    }


def farthest_point_indices(points, count):
    count = min(count, len(points))
    center = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - center) ** 2, axis=1)))]
    min_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.sum((points - points[next_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(selected, dtype=np.int64)


def pca_ratios(points):
    centered = points - points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    ratios = variance / max(variance.sum(), np.finfo(float).eps)
    participation = variance.sum() ** 2 / max(np.sum(variance**2), np.finfo(float).eps)
    return ratios, float(participation)


def local_pca_ratios(points, neighbors):
    neighbors = min(neighbors, len(points) - 1)
    distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    ratios = []
    for index in range(len(points)):
        local_indices = np.argpartition(distances[index], neighbors + 1)[1 : neighbors + 1]
        local = points[local_indices]
        ratios.append(pca_ratios(local)[0])
    return np.asarray(ratios)


def estimate_second_fundamental_form(points, point_index, g, gamma, neighbors):
    """Estimate the surface II from a local quadratic point-cloud patch."""
    neighbors = min(neighbors, len(points) - 1)
    center = points[point_index]
    delta_all = points - center
    distances = np.sum(delta_all**2, axis=1)
    local_indices = np.argpartition(distances, neighbors + 1)[1 : neighbors + 1]
    delta = delta_all[local_indices]

    covariance = delta.T @ delta / max(len(delta), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    tangent = eigenvectors[:, order[:2]]
    euclidean_normal = eigenvectors[:, order[2]]

    uv = delta @ tangent
    height = delta @ euclidean_normal
    u = uv[:, 0]
    v = uv[:, 1]
    design = np.column_stack(
        [
            np.ones_like(u),
            u,
            v,
            0.5 * u**2,
            u * v,
            0.5 * v**2,
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, height, rcond=None)
    fitted_height = design @ coefficients

    first_u = tangent[:, 0] + euclidean_normal * coefficients[1]
    first_v = tangent[:, 1] + euclidean_normal * coefficients[2]
    first = np.column_stack([first_u, first_v])
    second = np.empty((3, 2, 2), dtype=float)
    second[:, 0, 0] = euclidean_normal * coefficients[3]
    second[:, 0, 1] = euclidean_normal * coefficients[4]
    second[:, 1, 0] = second[:, 0, 1]
    second[:, 1, 1] = euclidean_normal * coefficients[5]

    metric_normal_constraint = first.T @ g
    _, _, vh = np.linalg.svd(metric_normal_constraint)
    metric_normal = vh[-1]
    metric_normal /= np.sqrt(metric_normal @ g @ metric_normal)

    induced_metric = first.T @ g @ first
    induced_inverse = np.linalg.inv(induced_metric)
    second_form = np.empty((2, 2), dtype=float)
    for first_index in range(2):
        for second_index in range(2):
            covariant_second = second[:, first_index, second_index].copy()
            covariant_second += np.einsum(
                "kij,i,j->k",
                gamma,
                first[:, first_index],
                first[:, second_index],
            )
            second_form[first_index, second_index] = metric_normal @ g @ covariant_second

    second_form_norm_sq = np.einsum(
        "ac,bd,ab,cd->",
        induced_inverse,
        induced_inverse,
        second_form,
        second_form,
    )
    mean_curvature = 0.5 * np.trace(induced_inverse @ second_form)
    local_metric_radius = np.median(np.sqrt(np.einsum("ni,ij,nj->n", delta, g, delta)))
    fit_rmse = np.sqrt(np.mean((height - fitted_height) ** 2))
    height_rms = np.sqrt(np.mean(height**2))

    local_variance = np.sort(eigenvalues)[::-1]
    local_variance_ratio = local_variance / max(local_variance.sum(), np.finfo(float).eps)
    return {
        "second_form": second_form,
        "second_form_norm": float(np.sqrt(max(second_form_norm_sq, 0.0))),
        "dimensionless_second_form_norm": float(
            np.sqrt(max(second_form_norm_sq, 0.0)) * local_metric_radius
        ),
        "mean_curvature": float(mean_curvature),
        "local_metric_radius": float(local_metric_radius),
        "quadratic_fit_relative_rmse": float(fit_rmse / max(height_rms, 1e-12)),
        "local_pca_ratio": local_variance_ratio,
    }


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "relative_std_to_rms": float(
            np.std(values) / max(np.sqrt(np.mean(values**2)), np.finfo(float).eps)
        ),
    }


def load_metric(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    latent_dim = int(config["latent_dim"])
    metric = MetricNetwork(latent_dim=latent_dim)
    state = checkpoint["geodesic_state_dict"]
    metric_state = {
        name[len("metric.") :]: value
        for name, value in state.items()
        if name.startswith("metric.")
    }
    metric.load_state_dict(metric_state)
    metric = metric.double().eval()
    return metric, config


def run_self_test():
    class FunctionalMetric(nn.Module):
        def __init__(self, function):
            super().__init__()
            self.function = function

        def forward(self, x):
            return torch.stack([self.function(point) for point in x])

    def euclidean(point):
        return torch.eye(3, dtype=point.dtype, device=point.device)

    def sphere_product(point):
        theta = point[0]
        return torch.diag(
            torch.stack([torch.ones_like(theta), torch.sin(theta) ** 2, torch.ones_like(theta)])
        )

    point = torch.tensor([1.2, 0.3, 0.2], dtype=torch.float64)
    flat = curvature_at(FunctionalMetric(euclidean), point)
    product = curvature_at(FunctionalMetric(sphere_product), point)
    assert abs(float(flat["scalar"])) < 1e-10
    assert np.allclose(flat["sectional_eigenvalues"].numpy(), 0.0, atol=1e-10)
    assert np.isclose(float(product["scalar"]), 2.0, atol=1e-8)
    assert np.allclose(
        product["sectional_eigenvalues"].numpy(),
        np.array([0.0, 0.0, 1.0]),
        atol=1e-8,
    )
    print("Curvature self-test passed: Euclidean R3 and S2 x R.")


def make_figure(out_path, trajectories, sample_indices, scalar, sectional, ii_norm):
    flat = trajectories.reshape(-1, 3)
    figure = plt.figure(figsize=(13, 10))
    axis = figure.add_subplot(2, 2, 1, projection="3d")
    for trajectory in trajectories:
        axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#777777", alpha=0.65)
    scatter = axis.scatter(
        flat[sample_indices, 0],
        flat[sample_indices, 1],
        flat[sample_indices, 2],
        c=scalar,
        cmap="coolwarm",
        s=28,
    )
    figure.colorbar(scatter, ax=axis, shrink=0.65, label="Scalar curvature")
    axis.set_title("Learned latent trajectories")
    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.set_zlabel("z3")

    axis = figure.add_subplot(2, 2, 2)
    axis.hist(scalar, bins=min(20, max(5, len(scalar) // 3)), color="#466C95")
    axis.set_title("Scalar curvature on trajectory support")
    axis.set_xlabel("Scalar curvature")
    axis.set_ylabel("Sample count")

    axis = figure.add_subplot(2, 2, 3)
    axis.boxplot(
        [sectional[:, index] for index in range(sectional.shape[1])],
        tick_labels=["K1", "K2", "K3"],
    )
    axis.axhline(0.0, color="#777777", linewidth=1)
    axis.set_title("Principal sectional curvatures")
    axis.set_xlabel("Curvature-operator eigenvalue")
    axis.set_ylabel("Sectional curvature")

    axis = figure.add_subplot(2, 2, 4)
    axis.hist(ii_norm, bins=min(20, max(5, len(ii_norm) // 3)), color="#D46A3A")
    axis.axvline(0.0, color="#777777", linewidth=1)
    axis.set_title("Estimated surface bending")
    axis.set_xlabel("Dimensionless ||II||")
    axis.set_ylabel("Sample count")

    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--max-curvature-points", type=int, default=48)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        if args.run_dir is None:
            return
    if args.run_dir is None:
        parser.error("run_dir is required unless only --self-test is used")

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = run_dir / "model_checkpoint.pt"
    outputs_path = run_dir / "fit_outputs.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist. Curvature requires the learned metric weights; "
            "rerun fit_sphere_trials.py with the current checkpoint-saving code."
        )
    if not outputs_path.exists():
        raise FileNotFoundError(outputs_path)

    metric, config = load_metric(checkpoint_path)
    if int(config["latent_dim"]) != 3:
        raise ValueError("This analysis currently expects latent_dim=3.")

    outputs = np.load(outputs_path)
    trajectories = np.asarray(outputs["latents_geo"], dtype=np.float64)
    points = trajectories.reshape(-1, 3)
    sample_indices = farthest_point_indices(points, args.max_curvature_points)

    metrics = []
    for output_index, point_index in enumerate(sample_indices, start=1):
        print(f"Curvature point {output_index}/{len(sample_indices)}", flush=True)
        point = torch.tensor(points[point_index], dtype=torch.float64)
        metrics.append(curvature_at(metric, point))

    metric_values = np.stack([item["metric"].numpy() for item in metrics])
    gamma_values = np.stack([item["gamma"].numpy() for item in metrics])
    scalar = np.asarray([float(item["scalar"]) for item in metrics])
    ricci_eigenvalues = np.stack([item["ricci_eigenvalues"].numpy() for item in metrics])
    sectional_eigenvalues = np.stack([item["sectional_eigenvalues"].numpy() for item in metrics])
    metric_eigenvalues = np.linalg.eigvalsh(metric_values)
    metric_condition = metric_eigenvalues[:, -1] / metric_eigenvalues[:, 0]

    global_ratios, global_participation = pca_ratios(points)
    local_ratios = local_pca_ratios(points, args.neighbors)
    surface_results = []
    for sample_position, point_index in enumerate(sample_indices):
        surface_results.append(
            estimate_second_fundamental_form(
                points,
                int(point_index),
                metric_values[sample_position],
                gamma_values[sample_position],
                args.neighbors,
            )
        )
    second_form_norm = np.asarray([item["second_form_norm"] for item in surface_results])
    dimensionless_second_form_norm = np.asarray(
        [item["dimensionless_second_form_norm"] for item in surface_results]
    )
    mean_curvature = np.asarray([item["mean_curvature"] for item in surface_results])
    surface_fit_error = np.asarray(
        [item["quadratic_fit_relative_rmse"] for item in surface_results]
    )

    curvature_rms = np.sqrt(np.mean(sectional_eigenvalues**2))
    constant_curvature_residual = np.sqrt(
        np.mean((sectional_eigenvalues - sectional_eigenvalues.mean()) ** 2)
    ) / max(curvature_rms, np.finfo(float).eps)
    local_third = local_ratios[:, 2]
    local_second = local_ratios[:, 1]

    report = {
        "run_dir": str(run_dir),
        "config": config,
        "sampling": {
            "trajectory_points": int(len(points)),
            "curvature_points": int(len(sample_indices)),
            "surface_neighbors": int(min(args.neighbors, len(points) - 1)),
        },
        "metric": {
            "eigenvalues": [numeric_summary(metric_eigenvalues[:, index]) for index in range(3)],
            "condition_number": numeric_summary(metric_condition),
        },
        "sectional_curvature": {
            "principal_eigenvalues": [
                numeric_summary(sectional_eigenvalues[:, index]) for index in range(3)
            ],
            "all": numeric_summary(sectional_eigenvalues.reshape(-1)),
            "constant_curvature_relative_residual": float(constant_curvature_residual),
        },
        "ricci_curvature": {
            "orthonormal_eigenvalues": [
                numeric_summary(ricci_eigenvalues[:, index]) for index in range(3)
            ]
        },
        "scalar_curvature": numeric_summary(scalar),
        "curvature_constancy": {
            "scalar_relative_std_to_rms": numeric_summary(scalar)["relative_std_to_rms"],
            "sectional_constant_curvature_relative_residual": float(
                constant_curvature_residual
            ),
            "approximately_constant_at_5_percent": bool(
                constant_curvature_residual < 0.05
            ),
        },
        "latent_surface": {
            "global_pca_variance_ratio": global_ratios.tolist(),
            "global_participation_ratio": global_participation,
            "local_pca_median_variance_ratio": np.median(local_ratios, axis=0).tolist(),
            "local_third_variance": numeric_summary(local_third),
            "local_second_variance": numeric_summary(local_second),
            "consistent_with_local_2d_surface": bool(
                np.median(local_third) < 0.01 and np.median(local_second) > 0.02
            ),
        },
        "second_fundamental_form": {
            "norm": numeric_summary(second_form_norm),
            "dimensionless_norm": numeric_summary(dimensionless_second_form_norm),
            "mean_curvature": numeric_summary(mean_curvature),
            "quadratic_surface_fit_relative_rmse": numeric_summary(surface_fit_error),
            "vanishes_numerically": bool(np.max(dimensionless_second_form_norm) < 1e-3),
            "note": (
                "II is estimated from local quadratic fits to a sparse collection of trajectories. "
                "It is diagnostic, not an exact surface calculation unless the trajectories densely "
                "sample a common smooth 2D surface."
            ),
        },
    }

    report_path = run_dir / "metric_geometry_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        run_dir / "metric_geometry_analysis.npz",
        sample_indices=sample_indices,
        sample_points=points[sample_indices],
        metric=metric_values,
        scalar_curvature=scalar,
        ricci_eigenvalues=ricci_eigenvalues,
        sectional_eigenvalues=sectional_eigenvalues,
        metric_eigenvalues=metric_eigenvalues,
        metric_condition_number=metric_condition,
        local_pca_variance_ratio=local_ratios,
        second_form_norm=second_form_norm,
        dimensionless_second_form_norm=dimensionless_second_form_norm,
        mean_curvature=mean_curvature,
        surface_fit_relative_rmse=surface_fit_error,
    )
    make_figure(
        run_dir / "metric_geometry_analysis.png",
        trajectories,
        sample_indices,
        scalar,
        sectional_eigenvalues,
        dimensionless_second_form_norm,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved metric analysis to {report_path}")


if __name__ == "__main__":
    main()
