#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"
uv pip install --python .venv/bin/python crocoddyl

echo
echo "Crocoddyl item 5 dependencies installed into $REPO_ROOT/.venv"
echo "Run item 5 with:"
echo "bash ./scripts/run_item5_crocoddyl.sh"
