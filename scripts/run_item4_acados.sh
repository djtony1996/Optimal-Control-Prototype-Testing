#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACADOS_ROOT="${ACADOS_SOURCE_DIR:-/Users/jitongding/Documents/GitHub/acados}"

export ACADOS_SOURCE_DIR="$ACADOS_ROOT"
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:$ACADOS_ROOT/lib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="$REPO_ROOT/src"

cd "$REPO_ROOT"
exec .venv/bin/python -m optimal_control_prototype_testing.acados_cpu "$@"
