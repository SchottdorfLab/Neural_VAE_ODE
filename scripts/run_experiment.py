#!/usr/bin/env python3
"""
Run a training script with a config, capture metadata, and snapshot artifacts.

Usage:
  python scripts/run_experiment.py --script src/v5_neural_vae.py --config configs/v5_base.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RUNS_DIR = ROOT / "runs"


def _run(cmd, cwd=None, check=True, input_text=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        input=input_text,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _try_git(args):
    try:
        return _run(["git"] + args, cwd=ROOT, check=True).stdout.strip()
    except Exception:
        return ""


def _summarize_stat(stat_text):
    stat_text = (stat_text or "").strip()
    if not stat_text:
        return ""
    lines = [l for l in stat_text.splitlines() if "|" in l]
    files = []
    for line in lines:
        path, rest = line.split("|", 1)
        path = path.strip()
        m = re.search(r"(\d+)", rest)
        if m:
            files.append(f"{path} ({m.group(1)} lines)")
        else:
            files.append(path)
    if not files:
        return ""
    if len(files) == 1:
        return f"Changed {files[0]}."
    suffix = "..." if len(files) > 5 else ""
    return f"Changed {len(files)} files: {', '.join(files[:5])}{suffix}."


def _get_previous_run(index_path):
    if not index_path.exists():
        return None
    with index_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def _write_config_with_out_dir(src_config_path, dst_config_path, out_dir):
    lines = src_config_path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key == "out_dir":
            out_lines.append(f"out_dir = {out_dir}")
            found = True
        else:
            out_lines.append(line)
    if not found:
        out_lines.append(f"out_dir = {out_dir}")
    dst_config_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _extract_r2(metrics):
    if not isinstance(metrics, dict):
        return ""
    for key in ("r2", "final_r2", "mean_r2"):
        if key in metrics:
            return metrics[key]
    return ""

def _to_jsonable(obj):
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
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
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Path to training script (e.g. src/v5_neural_vae.py)")
    parser.add_argument("--config", required=True, help="Path to config file (e.g. configs/v5_base.txt)")
    parser.add_argument("--run-id", default="", help="Optional run id (default: timestamp + short commit)")
    parser.add_argument("--note", default="", help="Optional note to include in run metadata")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    script_path = (ROOT / args.script).resolve()
    config_path = (ROOT / args.config).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_commit = _try_git(["rev-parse", "HEAD"]) or "unknown"
    git_short = _try_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    git_status = _try_git(["status", "--porcelain"])
    git_dirty = bool(git_status.strip())

    run_id = args.run_id or f"{timestamp}_{git_short}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_log_path = run_dir / "run.log"
    run_json_path = run_dir / "run.json"
    index_path = RUNS_DIR / "index.csv"

    # Write configs (original + resolved)
    shutil.copy2(config_path, run_dir / "config_original.txt")
    resolved_config = run_dir / "config.txt"
    _write_config_with_out_dir(config_path, resolved_config, str(run_dir))

    # Git diff snapshot (if dirty)
    diff_text = _try_git(["diff"])
    diff_cached_text = _try_git(["diff", "--cached"])
    diff_patch_path = ""
    if (diff_text or diff_cached_text) and git_dirty:
        diff_patch_path = str(run_dir / "git_diff.patch")
        (run_dir / "git_diff.patch").write_text(
            (diff_text or "") + ("\n" + diff_cached_text if diff_cached_text else ""),
            encoding="utf-8",
        )

    # Build code-change summary
    prev = _get_previous_run(index_path)
    code_summary = ""
    if prev and prev.get("commit") and git_commit != "unknown":
        prev_commit = prev["commit"]
        if prev_commit and prev_commit != git_commit:
            stat_since_prev = _try_git(["diff", f"{prev_commit}..{git_commit}", "--stat"])
            code_summary = _summarize_stat(stat_since_prev)
    if not code_summary:
        stat = _try_git(["diff", "--stat"])
        stat_cached = _try_git(["diff", "--cached", "--stat"])
        combined_stat = "\n".join([s for s in [stat, stat_cached] if s])
        code_summary = _summarize_stat(combined_stat)
    if not code_summary:
        code_summary = f"No uncommitted changes; code matches {git_short}."

    # If user provides a local LLM command, run it to override summary
    llm_cmd = os.environ.get("LLM_SUMMARY_CMD", "").strip()
    if llm_cmd and (diff_text or diff_cached_text):
        try:
            llm_args = llm_cmd.split()
            prompt = (
                "Summarize in one sentence the code changes in this diff. "
                "Focus on what behavior or features changed.\n\n"
                + (diff_text or "")
                + ("\n" + diff_cached_text if diff_cached_text else "")
            )
            llm_out = _run(llm_args, cwd=ROOT, check=True, input_text=prompt).stdout.strip()
            if llm_out:
                code_summary = llm_out.splitlines()[0].strip()
        except Exception:
            pass

    run_meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "script": str(script_path),
        "config": str(resolved_config),
        "config_original": str(run_dir / "config_original.txt"),
        "note": args.note,
        "git_commit": git_commit,
        "git_short": git_short,
        "git_dirty": git_dirty,
        "git_status": git_status,
        "git_diff_patch": diff_patch_path,
        "code_change_summary": code_summary,
    }

    # Execute training script
    with run_log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(script_path), "--config", str(resolved_config)],
            cwd=SRC_DIR,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    run_meta["exit_code"] = proc.returncode

    # Collect artifacts
    artifact_candidates = [
        SRC_DIR / "preview.png",
        SRC_DIR / "training_results.txt",
        SRC_DIR / "script_output.txt",
        SRC_DIR / "pt_files" / "ode_vae_best.pt",
        SRC_DIR / "pt_files" / "final_metrics.pt",
        SRC_DIR / "pt_files" / "trained_pca.pkl",
        SRC_DIR / "pt_files" / "latent_manifold_mds.png",
    ]
    artifacts = []
    for path in artifact_candidates:
        if path.exists() and path.is_file():
            dst = run_dir / path.name
            if dst.resolve() != path.resolve():
                try:
                    shutil.copy2(path, dst)
                    artifacts.append(str(dst))
                except Exception:
                    pass
    run_meta["artifacts"] = artifacts

    # Load metrics if available
    metrics = {}
    run_metadata_path = run_dir / "run_metadata.json"
    if run_metadata_path.exists():
        try:
            metrics = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    if not metrics:
        final_metrics_path = run_dir / "final_metrics.pt"
        if final_metrics_path.exists():
            try:
                import torch  # type: ignore
                metrics = torch.load(str(final_metrics_path), map_location="cpu")
            except Exception:
                metrics = {}
    run_meta["metrics"] = metrics

    # Write run.json
    run_json_path.write_text(json.dumps(_to_jsonable(run_meta), indent=2), encoding="utf-8")

    # Update index.csv
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id",
                "timestamp",
                "script",
                "config",
                "commit",
                "dirty",
                "exit_code",
                "r2",
                "note",
            ])
        writer.writerow([
            run_id,
            timestamp,
            str(script_path),
            str(resolved_config),
            git_commit,
            str(git_dirty),
            proc.returncode,
            _extract_r2(metrics),
            args.note,
        ])

    print(f"Run complete: {run_id}")
    print(f"run.json: {run_json_path}")
    print(f"index.csv: {index_path}")


if __name__ == "__main__":
    main()
