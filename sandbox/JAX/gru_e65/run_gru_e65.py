#!/usr/bin/env python3
"""Run the JAX E65 geodesic/free comparison with a GRU context encoder."""

import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPOSITORY_ROOT / "sandbox" / "JAX" / "tmp_e65.ipynb"

# Keep the GRU experiment isolated from the original flattened-MLP default.
os.environ["JAX_E65_ENCODER"] = "gru"
os.chdir(REPOSITORY_ROOT)
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_tmp_e65_notebook import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--notebook", str(NOTEBOOK_PATH)]
    main()
