#!/usr/bin/env bash

# Add the dependencies used by sandbox/JAX to the existing odevae environment.
#
# DARWIN (default):
#   bash scripts/install_jax_environment.sh
#
# Local macOS (JAX CPU; Apple GPU is not supported by upstream JAX):
#   JAX_PLATFORM=cpu bash scripts/install_jax_environment.sh

set -euo pipefail

PYTHON_BIN="${ODEVAE_PYTHON:-${HOME}/envs/odevae/bin/python}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda12}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at: $PYTHON_BIN" >&2
    echo "Set ODEVAE_PYTHON to the environment's Python executable." >&2
    exit 1
fi

case "$JAX_PLATFORM" in
    cuda12|cuda13)
        JAX_PACKAGE="jax[${JAX_PLATFORM}]"
        ;;
    cpu)
        JAX_PACKAGE="jax"
        ;;
    *)
        echo "JAX_PLATFORM must be cuda12, cuda13, or cpu; got: $JAX_PLATFORM" >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing into: $PYTHON_BIN"
echo "JAX platform: $JAX_PLATFORM"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --upgrade \
    "$JAX_PACKAGE" \
    -r "$REPO_ROOT/requirements-jax.txt"

if [[ "${INSTALL_JUPYTER:-false}" == "true" ]]; then
    "$PYTHON_BIN" -m pip install --upgrade jupyterlab ipykernel
fi

"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" - <<'PY'
import jax
import matplotlib
import numpy
import optax

print(f"jax:       {jax.__version__}")
print(f"jaxlib:    {jax.lib.__version__}")
print(f"optax:     {optax.__version__}")
print(f"numpy:     {numpy.__version__}")
print(f"matplotlib:{matplotlib.__version__}")
print(f"JAX devices: {jax.devices()}")
PY

echo "JAX dependencies installed successfully."
if [[ "$JAX_PLATFORM" == cuda12 || "$JAX_PLATFORM" == cuda13 ]]; then
    echo "A login node may show only a CPU. Verify the GPU inside a GPU allocation."
fi
