#!/usr/bin/env python3
"""
Restore a config file from the most recent run directory.

Default behavior:
- Finds the last run via runs/index.csv (fallback to newest run dir).
- Copies config_original.txt into a destination file.

Example:
  python scripts/restore_last_config.py --dest configs/v5_base.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
INDEX_PATH = RUNS_DIR / "index.csv"


def _get_last_run_dir(offset: int) -> Path | None:
    if INDEX_PATH.exists():
        with INDEX_PATH.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and offset <= len(rows):
            run_id = rows[-offset].get("run_id", "").strip()
            if run_id:
                run_dir = RUNS_DIR / run_id
                if run_dir.exists():
                    return run_dir
    # Fallback: newest run directory (exclude static output folder)
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        [
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name != "ode_vae_E65"
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    if offset <= len(candidates):
        return candidates[offset - 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default="configs/v5_base.txt",
        help="Destination config path to overwrite",
    )
    parser.add_argument(
        "--use-resolved",
        action="store_true",
        help="Copy config.txt (resolved) instead of config_original.txt",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Which run to restore from: 1=latest, 2=previous, etc.",
    )
    args = parser.parse_args()

    if args.offset < 1:
        print("--offset must be >= 1", file=sys.stderr)
        return 1

    run_dir = _get_last_run_dir(args.offset)
    if run_dir is None:
        print("No run directory found.", file=sys.stderr)
        return 1

    if args.use_resolved:
        src = run_dir / "config.txt"
    else:
        src = run_dir / "config_original.txt"
        if not src.exists():
            src = run_dir / "config.txt"

    if not src.exists():
        print(f"Config file not found in run dir: {run_dir}", file=sys.stderr)
        return 1

    dest_path = (ROOT / args.dest).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Restored {src} -> {dest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
