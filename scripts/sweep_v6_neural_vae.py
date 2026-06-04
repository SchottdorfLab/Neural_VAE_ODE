#!/usr/bin/env python3
"""Run v6_neural_vae sweeps over PCA dimension and raw/event loss settings.

Default behavior is a one-factor sweep around the base config. That keeps the
run count manageable while testing which axis improves held-out R2. Use
`--sweep-mode grid` for the full factorial combination of all supplied values.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v6_mind_geometry.txt"
DEFAULT_SCRIPT = ROOT / "src" / "v6_neural_vae.py"


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


def parse_list(raw: str) -> list[Any]:
    return [parse_scalar(x) for x in raw.split(",") if x.strip()]


def value_to_config(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def value_to_name(value: Any) -> str:
    text = value_to_config(value)
    text = text.replace(".", "p").replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_")


def read_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        cfg[key.strip()] = value.split("#", 1)[0].strip()
    return cfg


def write_config(base_path: Path, dst_path: Path, overrides: dict[str, Any]) -> None:
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in base_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            out_lines.append(f"{key} = {value_to_config(overrides[key])}")
            seen.add(key)
        else:
            out_lines.append(line)
    for key, value in overrides.items():
        if key not in seen:
            out_lines.append(f"{key} = {value_to_config(value)}")
    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_scalar(value)
    return overrides


def build_run_specs(args: argparse.Namespace, base_cfg: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    dimension_values = parse_list(args.dimensions)
    raw_values = parse_list(args.lambda_raw_recon)
    alpha_values = parse_list(args.event_weight_alpha)
    mode_values = parse_list(args.event_weight_mode)

    axes = [
        (args.dimension_key, dimension_values),
        ("lambda_raw_recon", raw_values),
        ("event_weight_alpha", alpha_values),
        ("event_weight_mode", mode_values),
    ]

    specs: list[tuple[str, dict[str, Any]]] = []
    if args.sweep_mode == "grid":
        for combo in itertools.product(*(values for _, values in axes)):
            overrides = {key: value for (key, _), value in zip(axes, combo)}
            name = "__".join(f"{key}_{value_to_name(value)}" for key, value in overrides.items())
            specs.append((name, overrides))
    else:
        specs.append(("base", {}))
        for key, values in axes:
            current = parse_scalar(base_cfg.get(key, ""))
            for value in values:
                if value == current:
                    continue
                else:
                    name = f"{key}_{value_to_name(value)}"
                specs.append((name, {key: value}))

    deduped: list[tuple[str, dict[str, Any]]] = []
    seen_names: dict[str, int] = {}
    for name, overrides in specs:
        count = seen_names.get(name, 0)
        seen_names[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        deduped.append((name, overrides))
    return deduped


def load_metrics(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = meta.get("metrics", {})
            return {"metadata": meta, "metrics": metrics}
        except Exception:
            pass
    final_path = run_dir / "final_metrics.pt"
    if final_path.exists():
        try:
            import torch  # type: ignore

            return {"metadata": {}, "metrics": torch.load(str(final_path), map_location="cpu")}
        except Exception:
            pass
    return {"metadata": {}, "metrics": {}}


def nested_get(data: dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def append_summary_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "run_name",
        "status",
        "exit_code",
        "run_dir",
        "config_path",
        "duration_sec",
        "dimension_key",
        "dimension_value",
        "lambda_raw_recon",
        "event_weight_alpha",
        "event_weight_mode",
        "heldout_R2",
        "heldout_MIND_R2",
        "heldout_r",
        "heldout_trial_median_MIND_R2",
        "teacher_heldout_R2",
        "event_frame_R2",
        "top_1_percent_event_capture",
        "pred_std_over_true_std",
        "pred_dynamics_std_over_true_dynamics_std",
        "pca_explained_variance",
        "best_epoch",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "v6_neural_sweeps")
    parser.add_argument("--sweep-name", default="")
    parser.add_argument("--sweep-mode", choices=["one_factor", "grid"], default="one_factor")
    parser.add_argument("--dimension-key", default="pca_dim", help="Usually pca_dim; can be latent_dim for a latent dimension sweep.")
    parser.add_argument("--dimensions", default="50,64,80,100,128")
    parser.add_argument("--lambda-raw-recon", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--event-weight-alpha", default="0,2,5,10")
    parser.add_argument("--event-weight-mode", default="dx,activity,activity_or_dx")
    parser.add_argument("--epochs", type=int, default=None, help="Optional quick-sweep epoch override.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N specs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="Extra config override, e.g. --set seed=7")
    args = parser.parse_args()

    base_config = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    script = args.script if args.script.is_absolute() else ROOT / args.script
    if not base_config.exists():
        raise FileNotFoundError(base_config)
    if not script.exists():
        raise FileNotFoundError(script)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_name = args.sweep_name or f"{stamp}_{args.sweep_mode}"
    sweep_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    sweep_root = sweep_root / sweep_name
    config_dir = sweep_root / "configs"
    log_dir = sweep_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_root / "summary.csv"
    plan_path = sweep_root / "plan.json"

    base_cfg = read_config(base_config)
    specs = build_run_specs(args, base_cfg)
    if args.limit is not None:
        specs = specs[: max(args.limit, 0)]
    extra_overrides = parse_set_overrides(args.set)

    plan = {
        "base_config": str(base_config),
        "script": str(script),
        "sweep_mode": args.sweep_mode,
        "dimension_key": args.dimension_key,
        "run_count": len(specs),
        "runs": [{"name": name, "overrides": overrides} for name, overrides in specs],
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Sweep root: {sweep_root}")
    print(f"Runs planned: {len(specs)}")
    print(f"Summary CSV: {summary_path}")
    if args.dry_run:
        for name, overrides in specs:
            print(f"DRY {name}: {overrides}")
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for idx, (name, overrides) in enumerate(specs, start=1):
        run_dir = sweep_root / "runs" / f"{idx:03d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{idx:03d}_{name}.txt"
        all_overrides: dict[str, Any] = {
            **overrides,
            **extra_overrides,
            "out_dir": run_dir,
        }
        if args.epochs is not None:
            all_overrides["epochs"] = args.epochs
        write_config(base_config, config_path, all_overrides)

        print(f"\n[{idx}/{len(specs)}] {name}")
        print(f"  overrides: {overrides}")
        start = time.time()
        log_path = log_dir / f"{idx:03d}_{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(script), "--config", str(config_path)],
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

        loaded = load_metrics(run_dir)
        metrics = loaded["metrics"]
        meta = loaded["metadata"]
        row = {
            "run_name": name,
            "status": "ok" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "run_dir": run_dir,
            "config_path": config_path,
            "duration_sec": f"{duration:.1f}",
            "dimension_key": args.dimension_key,
            "dimension_value": all_overrides.get(args.dimension_key, parse_scalar(base_cfg.get(args.dimension_key, ""))),
            "lambda_raw_recon": all_overrides.get("lambda_raw_recon", parse_scalar(base_cfg.get("lambda_raw_recon", ""))),
            "event_weight_alpha": all_overrides.get("event_weight_alpha", parse_scalar(base_cfg.get("event_weight_alpha", ""))),
            "event_weight_mode": all_overrides.get("event_weight_mode", parse_scalar(base_cfg.get("event_weight_mode", ""))),
            "heldout_R2": nested_get(metrics, "heldout.R2"),
            "heldout_MIND_R2": nested_get(metrics, "heldout.MIND_R2", nested_get(metrics, "heldout.variance_explained")),
            "heldout_r": nested_get(metrics, "heldout.r"),
            "heldout_trial_median_MIND_R2": nested_get(metrics, "heldout.MIND_R2_trial_median"),
            "teacher_heldout_R2": nested_get(metrics, "teacher_heldout.R2"),
            "event_frame_R2": nested_get(metrics, "events.event_frame_R2"),
            "top_1_percent_event_capture": nested_get(metrics, "events.top_1_percent_event_capture"),
            "pred_std_over_true_std": nested_get(metrics, "events.pred_std_over_true_std"),
            "pred_dynamics_std_over_true_dynamics_std": nested_get(metrics, "events.pred_dynamics_std_over_true_dynamics_std"),
            "pca_explained_variance": meta.get("pca_explained_variance", ""),
            "best_epoch": nested_get(meta, "training.best_epoch"),
        }
        append_summary_row(summary_path, row)
        print(
            "  result: "
            f"R2={row['heldout_R2']} | MIND_R2={row['heldout_MIND_R2']} | "
            f"event_R2={row['event_frame_R2']}"
        )
        if proc.returncode != 0 and args.stop_on_failure:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
