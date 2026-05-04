#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_VERSION="$("$REPO_ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
DEFAULT_CROCODDYL_PYTHON_PATH="$REPO_ROOT/.venv/lib/python${PY_VERSION}/site-packages/cmeel.prefix/lib/python${PY_VERSION}/site-packages"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="$REPO_ROOT/src"
export CROCODDYL_PYTHON_PATH="${CROCODDYL_PYTHON_PATH:-$DEFAULT_CROCODDYL_PYTHON_PATH}"

cd "$REPO_ROOT"
exec .venv/bin/python -m optimal_control_prototype_testing.crocoddyl_cpu
