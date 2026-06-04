#!/usr/bin/env python3
"""Benchmark model structures under the current strict heldout evaluation.

This compares older ODE/VAE/AE variants against the MIND-LLE and neural
geometric-AE models using matched trial construction and heldout scoring:

- position-aligned fixed-length trials
- globally inactive-neuron filtering
- train-only PCA feature space
- random 10% heldout trials by trial ID
- raw-space R2 after mapping reconstructions back to neuron space

The benchmark configs live in configs/model_structure_benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "model_structure_benchmarks"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    script: Path
    config: Path
    structure: str


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="v5_ode_vae",
        script=ROOT / "src" / "v5_neural_vae.py",
        config=CONFIG_DIR / "v5_ode_vae_strict.txt",
        structure="stochastic latent ODE-VAE",
    ),
    ModelSpec(
        name="v5_ode_ae",
        script=ROOT / "src" / "v5_neural_vae.py",
        config=CONFIG_DIR / "v5_ode_ae_strict.txt",
        structure="deterministic latent ODE-AE",
    ),
    ModelSpec(
        name="v5_ode_ae_reg",
        script=ROOT / "src" / "v5_neural_vae.py",
        config=CONFIG_DIR / "v5_ode_ae_reg_strict.txt",
        structure="deterministic latent ODE-AE + v5 regularizers",
    ),
    ModelSpec(
        name="v6_mind_lle",
        script=ROOT / "src" / "v6_mind_lle.py",
        config=CONFIG_DIR / "v6_mind_lle_strict.txt",
        structure="nonparametric MIND-LLE",
    ),
    ModelSpec(
        name="v6_plain_ae",
        script=ROOT / "src" / "v6_neural_vae.py",
        config=CONFIG_DIR / "v6_plain_ae_strict.txt",
        structure="deterministic PCA-feature AE",
    ),
    ModelSpec(
        name="v6_geometric_ae",
        script=ROOT / "src" / "v6_neural_vae.py",
        config=CONFIG_DIR / "v6_geometric_ae_strict.txt",
        structure="MIND-informed neural geometric AE",
    ),
)


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def value_to_config(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def read_config(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.split("#", 1)[0].strip()
    return out


def write_config(base_path: Path, dst_path: Path, overrides: dict[str, Any]) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for line in base_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            lines.append(f"{key} = {value_to_config(overrides[key])}")
            seen.add(key)
        else:
            lines.append(line)
    for key, value in overrides.items():
        if key not in seen:
            lines.append(f"{key} = {value_to_config(value)}")
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_set(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_scalar(value)
    return overrides


def nested_get(data: Any, path: str, default: Any = "") -> Any:
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def load_run_outputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    final_path = run_dir / "final_metrics.pt"
    if final_path.exists():
        try:
            import torch  # type: ignore

            loaded = torch.load(str(final_path), map_location="cpu")
            if isinstance(loaded, dict):
                metrics = to_jsonable(loaded)
        except Exception:
            metrics = {}

    if not metrics:
        metrics = metadata.get("metrics", {})
    return metadata, metrics


def extract_strict_metrics(metadata: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    # v6_mind_lle/v6_neural_vae shape.
    heldout = metrics.get("heldout", {}) if isinstance(metrics, dict) else {}
    events = metrics.get("events", {}) if isinstance(metrics, dict) else {}
    teacher = metrics.get("teacher_heldout", {}) if isinstance(metrics, dict) else {}

    strict_r2 = nested_get(metrics, "heldout.R2", "")
    strict_mind_r2 = nested_get(metrics, "heldout.MIND_R2", nested_get(metrics, "heldout.variance_explained", ""))
    strict_r = nested_get(metrics, "heldout.r", "")

    # v5 run_metadata/final_metrics shape.
    if strict_r2 == "":
        strict_r2 = metrics.get("r2", "")
    if strict_r == "":
        strict_r = metrics.get("r", "")
    if strict_r2 == "":
        strict_r2 = nested_get(metadata, "final_metrics.final_r2", "")
    if strict_r == "":
        strict_r = nested_get(metadata, "final_metrics.final_r", "")

    return {
        "strict_R2": strict_r2,
        "strict_MIND_R2": strict_mind_r2,
        "strict_r": strict_r,
        "heldout_trial_median_MIND_R2": heldout.get("MIND_R2_trial_median", ""),
        "teacher_heldout_R2": teacher.get("R2", ""),
        "event_frame_R2": events.get("event_frame_R2", ""),
        "top_1_percent_event_capture": events.get("top_1_percent_event_capture", ""),
        "pred_std_over_true_std": events.get("pred_std_over_true_std", ""),
        "pred_dynamics_std_over_true_dynamics_std": events.get("pred_dynamics_std_over_true_dynamics_std", ""),
        "n_trials_total": metadata.get("n_trials_total", ""),
        "n_trials_train": metadata.get("n_trials_train", ""),
        "n_trials_heldout": metadata.get("n_trials_heldout", ""),
        "heldout_fraction": metadata.get("heldout_fraction", ""),
        "feature_dim": metadata.get("feature_dim", ""),
        "latent_dim": metadata.get("latent_dim", ""),
        "pca_explained_variance": metadata.get("pca_explained_variance", ""),
        "best_epoch": nested_get(metadata, "training.best_epoch", ""),
    }


def selected_specs(names: str) -> list[ModelSpec]:
    if names.strip().lower() in {"", "all"}:
        return list(SPECS)
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    by_name = {spec.name: spec for spec in SPECS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"Unknown model name(s): {', '.join(missing)}. Known: {', '.join(by_name)}")
    return [spec for spec in SPECS if spec.name in wanted]


def append_row(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "model",
        "structure",
        "status",
        "exit_code",
        "duration_sec",
        "script",
        "config",
        "run_dir",
        "strict_R2",
        "strict_MIND_R2",
        "strict_r",
        "heldout_trial_median_MIND_R2",
        "teacher_heldout_R2",
        "event_frame_R2",
        "top_1_percent_event_capture",
        "pred_std_over_true_std",
        "pred_dynamics_std_over_true_dynamics_std",
        "n_trials_total",
        "n_trials_train",
        "n_trials_heldout",
        "heldout_fraction",
        "feature_dim",
        "latent_dim",
        "pca_explained_variance",
        "best_epoch",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="Comma-separated model names, or all.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "model_structure_benchmarks")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for configs that have an epochs key.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="Extra config override applied to every run.")
    args = parser.parse_args()

    specs = selected_specs(args.models)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_name = args.benchmark_name or stamp
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    bench_root = out_root / benchmark_name
    config_out = bench_root / "configs"
    log_out = bench_root / "logs"
    run_out = bench_root / "runs"
    config_out.mkdir(parents=True, exist_ok=True)
    log_out.mkdir(parents=True, exist_ok=True)
    run_out.mkdir(parents=True, exist_ok=True)
    summary_path = bench_root / "summary.csv"

    extra = parse_set(args.set)
    print(f"Benchmark root: {bench_root}")
    print(f"Models: {', '.join(spec.name for spec in specs)}")
    print(f"Summary: {summary_path}")

    if args.dry_run:
        for spec in specs:
            print(f"DRY {spec.name}: {spec.script} --config {spec.config}")
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for idx, spec in enumerate(specs, start=1):
        if not spec.script.exists():
            raise FileNotFoundError(spec.script)
        if not spec.config.exists():
            raise FileNotFoundError(spec.config)

        run_dir = run_out / f"{idx:02d}_{safe_name(spec.name)}"
        cfg_path = config_out / f"{idx:02d}_{safe_name(spec.name)}.txt"
        overrides: dict[str, Any] = {"out_dir": run_dir, **extra}
        if args.epochs is not None and "epochs" in read_config(spec.config):
            overrides["epochs"] = args.epochs
        write_config(spec.config, cfg_path, overrides)

        if args.skip_existing and (run_dir / "final_metrics.pt").exists():
            metadata, metrics = load_run_outputs(run_dir)
            row = {
                "model": spec.name,
                "structure": spec.structure,
                "status": "skipped_existing",
                "exit_code": 0,
                "duration_sec": "",
                "script": spec.script,
                "config": cfg_path,
                "run_dir": run_dir,
                **extract_strict_metrics(metadata, metrics),
            }
            append_row(summary_path, row)
            print(f"[{idx}/{len(specs)}] skipped existing {spec.name}")
            continue

        print(f"\n[{idx}/{len(specs)}] {spec.name}: {spec.structure}")
        print(f"  config: {cfg_path}")
        print(f"  out_dir: {run_dir}")
        start = time.time()
        log_path = log_out / f"{idx:02d}_{safe_name(spec.name)}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(spec.script), "--config", str(cfg_path)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        duration = time.time() - start

        metadata, metrics = load_run_outputs(run_dir)
        row = {
            "model": spec.name,
            "structure": spec.structure,
            "status": "ok" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "duration_sec": f"{duration:.1f}",
            "script": spec.script,
            "config": cfg_path,
            "run_dir": run_dir,
            **extract_strict_metrics(metadata, metrics),
        }
        append_row(summary_path, row)
        print(
            "  result: "
            f"R2={row['strict_R2']} | MIND_R2={row['strict_MIND_R2']} | "
            f"event_R2={row['event_frame_R2']}"
        )
        if proc.returncode != 0 and args.stop_on_failure:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
