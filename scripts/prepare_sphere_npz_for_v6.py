#!/usr/bin/env python3
"""Convert synthetic sphere-trial outputs into the E65-style NPZ expected by v6.

The v6 June-3 baseline expects frame-wise arrays:

    roi      [frames, neurons]
    Trial    [frames]
    Time     [frames]
    Position [frames]

The sphere scripts save trial-wise arrays:

    rates or activities [trials, time, neurons]

This adapter preserves the neural activity values and uses normalized within-trial
trajectory progress as the Position axis, so the original position-binned v6
configuration can run without changes to the model code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def first_available(data: np.lib.npyio.NpzFile, names: list[str]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of {names} found in {data.filename}; keys={data.files}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="synthetic_sphere_trials.npz from fit_sphere_trials.py")
    parser.add_argument("--output", required=True, type=Path, help="E65-style output NPZ for v6")
    parser.add_argument(
        "--position-mode",
        default="progress",
        choices=["progress", "time"],
        help="Position axis to expose to v6. progress maps every trial to [0,1].",
    )
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=True)
    rates = np.asarray(first_available(source, ["rates", "activities"]), dtype=np.float32)
    if rates.ndim != 3:
        raise ValueError(f"Expected rates/activities with shape [trials,time,neurons], got {rates.shape}")

    n_trials, n_time, n_neurons = rates.shape
    t_eval = np.asarray(source["t_eval"], dtype=np.float32) if "t_eval" in source.files else np.arange(n_time, dtype=np.float32)
    if t_eval.shape[0] != n_time:
        raise ValueError(f"t_eval length {t_eval.shape[0]} does not match trial length {n_time}")

    roi = rates.reshape(n_trials * n_time, n_neurons).astype(np.float32)
    trial = np.repeat(np.arange(n_trials, dtype=np.int64), n_time)
    time = np.tile(t_eval, n_trials).astype(np.float32)

    if args.position_mode == "time":
        base_position = t_eval.astype(np.float32)
    else:
        base_position = np.linspace(0.0, 1.0, n_time, dtype=np.float32)
    position = np.tile(base_position, n_trials).astype(np.float32)

    out_kwargs = {
        "roi": roi,
        "Trial": trial,
        "Time": time,
        "Position": position,
        "source_rates_shape": np.asarray(rates.shape, dtype=np.int64),
        "source_file": np.asarray(str(args.input)),
        "position_mode": np.asarray(args.position_mode),
    }
    for key in ["latents_theta_phi", "true_latents", "initial_conditions", "theta_centers", "phi_centers", "t_eval"]:
        if key in source.files:
            out_kwargs[key] = source[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out_kwargs)
    print(f"Wrote {args.output}")
    print(f"roi={roi.shape}, Trial={trial.shape}, Time={time.shape}, Position={position.shape}")


if __name__ == "__main__":
    main()
