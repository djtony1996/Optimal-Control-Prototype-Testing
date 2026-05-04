#!/usr/bin/env bash
set -euo pipefail

# Intended helper for item 4 (CPU reference baseline).
# This script documents the build flow without mixing acados setup into the
# JAX prototype environment.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /absolute/path/to/acados"
  exit 1
fi

ACADOS_ROOT="$1"

echo "Installing recommended macOS build tools for acados..."
brew install cmake gcc

export CC="$(brew --prefix)/bin/gcc-15"
export CXX="$(brew --prefix)/bin/g++-15"

echo "Using CC=$CC"
echo "Using CXX=$CXX"

mkdir -p "$ACADOS_ROOT/build"
cd "$ACADOS_ROOT/build"

cmake -DACADOS_WITH_OPENMP=ON -DBUILD_SHARED_LIBS=ON -DACADOS_WITH_QPOASES=OFF ..
make install -j"$(sysctl -n hw.ncpu)"

mkdir -p "$ACADOS_ROOT/bin"
curl -L \
  https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-osx-arm64 \
  -o "$ACADOS_ROOT/bin/t_renderer"
chmod +x "$ACADOS_ROOT/bin/t_renderer"

echo
echo "Next steps:"
echo "export ACADOS_SOURCE_DIR=\"$ACADOS_ROOT\""
echo "export DYLD_LIBRARY_PATH=\"\$DYLD_LIBRARY_PATH:$ACADOS_ROOT/lib\""
echo "export MPLCONFIGDIR=/tmp/matplotlib"
echo "uv pip install --python .venv/bin/python -e \"$ACADOS_ROOT/interfaces/acados_template\""
