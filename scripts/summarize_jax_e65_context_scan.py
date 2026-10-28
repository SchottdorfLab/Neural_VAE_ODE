#!/usr/bin/env python3
"""Print a compact comparison of completed JAX E65 context-scan runs."""

import json
from pathlib import Path


def main():
    summaries = []
    pattern = "jax_e65_context_scan_n*_c*_e*/summary.json"
    for path in Path("runs").glob(pattern):
        payload = json.loads(path.read_text())
        config = payload["config"]
        summaries.append(
            (
                int(config["context_steps"]),
                int(config["num_trials"]),
                int(config["epochs"]),
                payload["geodesic"]["heldout_context"]["r2"],
                payload["geodesic"]["forecast"]["r2"],
                payload["free"]["heldout_context"]["r2"],
                payload["free"]["forecast"]["r2"],
                path.parent,
            )
        )

    if not summaries:
        raise SystemExit(f"No completed summaries matched runs/{pattern}")

    print(
        f"{'ctx':>4} {'trials':>6} {'epochs':>6} "
        f"{'geo ctx R2':>11} {'geo fc R2':>11} "
        f"{'free ctx R2':>12} {'free fc R2':>12}  run"
    )
    for row in sorted(summaries):
        context, trials, epochs, geo_ctx, geo_fc, free_ctx, free_fc, run = row
        print(
            f"{context:4d} {trials:6d} {epochs:6d} "
            f"{geo_ctx:11.4f} {geo_fc:11.4f} "
            f"{free_ctx:12.4f} {free_fc:12.4f}  {run}"
        )


if __name__ == "__main__":
    main()
