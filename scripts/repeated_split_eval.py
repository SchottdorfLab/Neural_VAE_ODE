#!/usr/bin/env python3
"""Run repeated heldout-trial splits and aggregate reconstruction metrics.

This is meant for the paper-style question: how well does the reconstruction
hold up averaged across multiple random 10% heldout trial splits?
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_seed_list(raw: str | None, n: int, start: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [start + i for i in range(n)]


def write_config_with_overrides(src: Path, dst: Path, overrides: dict[str, Any]) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    found = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            out.append(f"{key} = {overrides[key]}")
            found.add(key)
        else:
            out.append(line)
    for key, value in overrides.items():
        if key not in found:
            out.append(f"{key} = {value}")
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def nested_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def row_from_metadata(split_seed: int, run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    metrics = meta.get("metrics", {})
    return {
        "split_seed": split_seed,
        "run_dir": str(run_dir),
        "raw_R2": nested_get(metrics, "raw_mapped_heldout.R2"),
        "raw_MIND_R2": nested_get(metrics, "raw_mapped_heldout.MIND_R2"),
        "raw_r": nested_get(metrics, "raw_mapped_heldout.r"),
        "raw_trial_mean": nested_get(metrics, "raw_mapped_heldout.MIND_R2_trial_mean"),
        "raw_trial_median": nested_get(metrics, "raw_mapped_heldout.MIND_R2_trial_median"),
        "raw_trial_min": nested_get(metrics, "raw_mapped_heldout.MIND_R2_trial_min"),
        "raw_trial_max": nested_get(metrics, "raw_mapped_heldout.MIND_R2_trial_max"),
        "binned_R2": nested_get(metrics, "binned_heldout.R2"),
        "binned_MIND_R2": nested_get(metrics, "binned_heldout.MIND_R2"),
        "binned_trial_median": nested_get(metrics, "binned_heldout.MIND_R2_trial_median"),
        "event_top1": nested_get(metrics, "raw_mapped_events.top_1_percent_event_capture"),
        "event_top0_5": nested_get(metrics, "raw_mapped_events.top_0_5_percent_event_capture"),
        "std_ratio": nested_get(metrics, "raw_mapped_events.pred_std_over_true_std"),
        "dyn_std_ratio": nested_get(metrics, "raw_mapped_events.pred_dynamics_std_over_true_dynamics_std"),
        "n_trials_total": meta.get("n_trials_total"),
        "n_trials_train": meta.get("n_trials_train"),
        "n_trials_heldout": meta.get("n_trials_heldout"),
        "n_heldout_raw_frames": meta.get("n_heldout_raw_frames"),
        "feature_dim": meta.get("feature_dim"),
        "latent_dim": meta.get("latent_dim"),
        "pca_explained_variance": meta.get("pca_explained_variance"),
    }


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    import numpy as np

    skip = {"split_seed", "run_dir"}
    keys = [k for k in rows[0].keys() if k not in skip]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = []
        for row in rows:
            val = row.get(key)
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                pass
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v6_binned_raw_eval_ae.txt")
    parser.add_argument("--script", default="src/v6_binned_raw_eval_ae.py")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--split-seeds", default="", help="Comma-separated split seeds. Overrides --splits/--seed-start.")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--epochs", default="", help="Optional epoch override for quick tests.")
    parser.add_argument("--bernoulli-split", action="store_true", help="Use MATLAB-style rand > frac splits instead of exact heldout counts.")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    script_path = (ROOT / args.script).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    split_seeds = parse_seed_list(args.split_seeds, args.splits, args.seed_start)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / f"repeated_split_eval_{timestamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, split_seed in enumerate(split_seeds, start=1):
        run_dir = out_dir / f"split_{i:02d}_seed_{split_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.txt"
        overrides: dict[str, Any] = {
            "out_dir": run_dir,
            "mind_split_seed": split_seed,
            "split_exact_test_fraction": str(not args.bernoulli_split).lower(),
        }
        if args.epochs:
            overrides["epochs"] = args.epochs
        write_config_with_overrides(config_path, cfg_path, overrides)

        print(f"\n=== Split {i}/{len(split_seeds)}: mind_split_seed={split_seed} ===", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--config", str(cfg_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        log_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_lines.append(line)
        proc.wait()
        (run_dir / "outer_run.log").write_text("".join(log_lines), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Split {split_seed} failed with exit code {proc.returncode}")

        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing run metadata: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        row = row_from_metadata(split_seed, run_dir, meta)
        rows.append(row)
        print(
            f"Split {split_seed} raw R2={row['raw_R2']:.4f} | "
            f"MIND_R2={row['raw_MIND_R2']:.4f} | trial median={row['raw_trial_median']:.4f}",
            flush=True,
        )

    summary = numeric_summary(rows)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": str(config_path),
        "script": str(script_path),
        "split_seeds": split_seeds,
        "rows": rows,
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    raw = summary.get("raw_R2", {})
    raw_mind = summary.get("raw_MIND_R2", {})
    trial_med = summary.get("raw_trial_median", {})
    print("\n=== Repeated split summary ===")
    print(f"out_dir: {out_dir}")
    print(f"raw R2 mean={raw.get('mean', float('nan')):.4f} std={raw.get('std', float('nan')):.4f}")
    print(f"raw MIND_R2 mean={raw_mind.get('mean', float('nan')):.4f} std={raw_mind.get('std', float('nan')):.4f}")
    print(f"raw trial median mean={trial_med.get('mean', float('nan')):.4f} std={trial_med.get('std', float('nan')):.4f}")
    print(f"summary.csv: {csv_path}")
    print(f"summary.json: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
