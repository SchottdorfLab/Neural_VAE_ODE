import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
DEFAULT_DATA = os.path.join(SRC_DIR, "npz_e65_data", "E65_data.npz")


def latest_run_dir(runs_dir):
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "analysis_cache_best.npz")):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No run directory with analysis_cache_best.npz was found.")
    return max(candidates, key=os.path.getmtime)


def flatten_modeled_block(x):
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    raise ValueError(f"Expected 2D or 3D cache arrays, got shape {x.shape}")


def filter_inactive_neurons_mind(roi):
    """
    Match the MIND Matlab rule:
        Neurons = sum(ROIactivities,1) > 0
    with roi in [T, N] format.
    """
    active_mask = roi.sum(axis=0) > 0
    return roi[:, active_mask], active_mask


def save_heatmap_pair(x_left, x_right, out_path, left_title, right_title):
    vmin = min(float(np.min(x_left)), float(np.min(x_right)))
    vmax = max(float(np.max(x_left)), float(np.max(x_right)))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(x_left.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(left_title)
    plt.xlabel("Time bins")
    plt.ylabel("Neuron")

    plt.subplot(1, 2, 2)
    plt.imshow(x_right.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(right_title)
    plt.xlabel("Time bins")
    plt.ylabel("Neuron")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate full-session and modeled heatmaps for a v5 run.")
    ap.add_argument("--run-dir", default=None, help="Run directory containing analysis_cache_best.npz")
    ap.add_argument("--data", default=DEFAULT_DATA, help="Path to raw E65_data.npz")
    args = ap.parse_args()

    runs_dir = os.path.join(ROOT_DIR, "runs")
    run_dir = args.run_dir if args.run_dir else latest_run_dir(runs_dir)
    out_dir = run_dir

    npz = np.load(args.data)
    roi_raw = npz["roi"]
    if roi_raw.shape[0] < roi_raw.shape[1]:
        roi_raw = roi_raw.T
    roi_raw_filtered, active_mask = filter_inactive_neurons_mind(roi_raw)
    silent_count = int((~active_mask).sum())
    active_count = int(active_mask.sum())

    raw_out = os.path.join(out_dir, "raw_full_session_heatmap.png")
    plt.figure(figsize=(14, 6))
    plt.imshow(roi_raw_filtered.T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.title(
        f"Full Raw Session Heatmap ({active_count} active neurons x {roi_raw_filtered.shape[0]} timepoints; "
        f"{silent_count} silent removed)"
    )
    plt.xlabel("Original timepoints")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(raw_out, dpi=160)
    plt.close()
    print(f"wrote {raw_out}")
    print(f"MIND inactive-neuron filter removed {silent_count} globally silent neurons; kept {active_count}.")

    cache_path = os.path.join(run_dir, "analysis_cache_best.npz")
    if not os.path.exists(cache_path):
        print(f"analysis cache not found at {cache_path}")
        return

    cache = np.load(cache_path, allow_pickle=True)
    x_true = flatten_modeled_block(cache["x_true"])
    x_pred = flatten_modeled_block(cache["x_pred"])
    if x_true.shape[1] == active_mask.shape[0]:
        x_true = x_true[:, active_mask]
    if x_pred.shape[1] == active_mask.shape[0]:
        x_pred = x_pred[:, active_mask]
    modeled_out = os.path.join(out_dir, "raw_vs_recon_heatmap.png")
    save_heatmap_pair(
        x_true,
        x_pred,
        modeled_out,
        f"Modeled Ground Truth ({x_true.shape[1]} neurons x {x_true.shape[0]} bins)",
        f"Modeled Reconstruction ({x_pred.shape[1]} neurons x {x_pred.shape[0]} bins)",
    )
    print(f"wrote {modeled_out}")


if __name__ == "__main__":
    main()
