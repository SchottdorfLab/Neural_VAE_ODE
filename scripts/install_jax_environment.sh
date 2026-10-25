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
JAX_VERSION="${JAX_VERSION:-0.4.28}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at: $PYTHON_BIN" >&2
    echo "Set ODEVAE_PYTHON to the environment's Python executable." >&2
    exit 1
fi

case "$JAX_PLATFORM" in
    cuda12)
        JAX_PACKAGE="jax[cuda12]==${JAX_VERSION}"
        CONSTRAINT_ARGS=(--constraint "$REPO_ROOT/constraints-jax-darwin.txt")
        ;;
    cpu)
        JAX_PACKAGE="jax==${JAX_VERSION}"
        CONSTRAINT_ARGS=()
        ;;
    *)
        echo "JAX_PLATFORM must be cuda12 or cpu; got: $JAX_PLATFORM" >&2
        exit 1
        ;;
esac

echo "Installing into: $PYTHON_BIN"
echo "JAX platform: $JAX_PLATFORM"
echo "JAX version: $JAX_VERSION"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install \
    --only-binary=:all: \
    --upgrade-strategy=only-if-needed \
    "${CONSTRAINT_ARGS[@]}" \
    "$JAX_PACKAGE" \
    -r "$REPO_ROOT/requirements-jax.txt"

if [[ "${INSTALL_JUPYTER:-false}" == "true" ]]; then
    "$PYTHON_BIN" -m pip install --upgrade jupyterlab ipykernel
fi

if ! "$PYTHON_BIN" -m pip check; then
    echo "Warning: pip reported a dependency conflict in the pre-existing environment." >&2
fi

JAX_PLATFORMS=cpu "$PYTHON_BIN" - <<'PY'
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
if [[ "$JAX_PLATFORM" == cuda12 ]]; then
    echo "A login node may show only a CPU. Verify the GPU inside a GPU allocation."
fi
