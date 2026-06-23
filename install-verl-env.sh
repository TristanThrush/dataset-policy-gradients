#!/bin/bash
#
# Build the verl PPO trainer environment from scratch (Python 3.10) using uv.
# Installs the pinned deps from requirements-verl.txt, then the local ./verl
# package WITHOUT its (stale) dependency pins.
#
# Usage: ./install-verl-env.sh [venv_path]   (default: .venv-verl)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=${1:-.venv-verl}
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

echo "Creating venv at $VENV (Python 3.10)..."
uv venv --python 3.10 "$VENV"

# requirements-verl.txt is a COMPLETE freeze (every transitive dep pinned), so we
# install with --no-deps: no resolution, just the exact versions. This is required
# because the source env is internally inconsistent (e.g. opentelemetry-sdk 1.36.0
# alongside an exporter that asks for <1.27) and any resolver would reject it.
# setuptools/wheel are omitted by `pip freeze`, and a fresh uv venv (unlike conda)
# doesn't ship them -- but verl imports pkg_resources (from setuptools) at startup.
echo "Seeding setuptools/wheel..."
uv pip install --python "$VENV" setuptools==78.1.1 wheel==0.45.1

echo "Installing pinned dependencies (--no-deps)..."
uv pip install --python "$VENV" --no-deps -r requirements-verl.txt

echo "Installing local editable package (--no-deps)..."
uv pip install --python "$VENV" --no-deps -e ./verl

echo "Done. Verifying imports..."
"$VENV/bin/python" -c "import torch, transformers, ray, verl; print('OK torch', torch.__version__, 'verl', getattr(verl, '__version__', '?'))"
