#!/usr/bin/env python3
"""Execute tmp_e65.ipynb cell-by-cell while streaming output to the terminal."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("sandbox/JAX/tmp_e65.ipynb"),
    )
    arguments = parser.parse_args()

    notebook_path = arguments.notebook.resolve()
    notebook = json.loads(notebook_path.read_text())
    namespace = {
        "__file__": str(notebook_path),
        "__name__": "__main__",
    }

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        print(f"\n--- executing notebook cell {index} ---", flush=True)
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
