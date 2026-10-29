#!/usr/bin/env python3
"""Summarize completed GRU E65 context-length runs."""

import json
from pathlib import Path


def main():
    rows = []
    pattern = "jax_e65_gru_context_scan_n*_c*_e*/summary.json"
    for path in Path("runs").glob(pattern):
        result = json.loads(path.read_text())
        config = result["config"]
        rows.append(
            (
                int(config["context_steps"]),
                int(config["geodesic_parameters"]),
                result["geodesic"]["heldout_context"]["r2"],
                result["geodesic"]["forecast"]["r2"],
                result["free"]["heldout_context"]["r2"],
                result["free"]["forecast"]["r2"],
                path.parent,
            )
        )

    if not rows:
        raise SystemExit(f"No completed summaries matched runs/{pattern}")

    print(
        f"{'ctx':>4} {'parameters':>10} {'geo ctx R2':>11} "
        f"{'geo fc R2':>11} {'free ctx R2':>12} {'free fc R2':>12}  run"
    )
    for context, parameters, geo_ctx, geo_fc, free_ctx, free_fc, run in sorted(rows):
        print(
            f"{context:4d} {parameters:10d} {geo_ctx:11.4f} "
            f"{geo_fc:11.4f} {free_ctx:12.4f} {free_fc:12.4f}  {run}"
        )


if __name__ == "__main__":
    main()
