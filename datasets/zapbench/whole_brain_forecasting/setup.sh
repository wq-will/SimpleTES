#!/usr/bin/env bash
set -euo pipefail

# Set up the ZAPBench environment and data cache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ZAPBENCH_SRC="$REPO_ROOT/third_party/zapbench"

if [ ! -d "$ZAPBENCH_SRC" ]; then
  echo "ERROR: $ZAPBENCH_SRC not found. Clone first:" >&2
  echo "  git -C \"$REPO_ROOT/third_party\" clone https://github.com/google-research/zapbench.git" >&2
  echo "  git -C \"$ZAPBENCH_SRC\" checkout b08d584e3ba80125788ac915eec63a2e4e11467b" >&2
  exit 1
fi

PYTHON_SPEC="${PYTHON_BIN:-}"

if [ -z "${PIP_INDEX_URL:-}" ]; then
  PIP_INDEX_URL="https://pypi.org/simple"
  export PIP_INDEX_URL
fi

echo "============================================"
echo "  Setting up ZAPBench task venv"
echo "============================================"
echo "Venv:        $VENV_DIR"
echo "zapbench:    $ZAPBENCH_SRC"
echo ""

verify_python() {
  "$VENV_DIR/bin/python" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):
    raise SystemExit(
        f"ERROR: Need Python >=3.11 in venv, got {sys.version.split()[0]}."
    )
print("Python OK:", sys.version.split()[0])
PY
}

if command -v uv >/dev/null 2>&1; then
  echo "[1/4] Using uv for venv + installs..."
  : "${UV_CACHE_DIR:=/tmp/uv-cache-zapbench}"
  export UV_CACHE_DIR
  export UV_DEFAULT_INDEX="$PIP_INDEX_URL"

  if [ -z "$PYTHON_SPEC" ]; then PYTHON_SPEC=">=3.11"; fi

  if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python "$PYTHON_SPEC" --prompt zapbench
  fi
  verify_python
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv pip install --upgrade pip setuptools wheel
else
  echo "[1/4] uv not found; falling back to python -m venv..."
  if [ -z "$PYTHON_SPEC" ]; then PYTHON_SPEC="python3"; fi
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_SPEC" -m venv "$VENV_DIR" --prompt zapbench
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  verify_python
  python -m pip install --upgrade pip setuptools wheel
fi

echo ""
echo "[2/4] Installing core deps (numpy/scipy/sklearn/pandas/tensorstore)..."
pip install --upgrade \
  numpy scipy scikit-learn scikit-image pandas einops tqdm \
  tensorstore

echo ""
echo "[3/4] Installing JAX (CUDA) and PyTorch (CUDA)..."
# JAX with CUDA 12 wheels. If your CUDA differs, override JAX_CUDA below.
JAX_CUDA="${JAX_CUDA:-cuda12}"
pip install "jax[$JAX_CUDA]" flax optax chex

# PyTorch CUDA wheels. Match the host CUDA driver version.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
pip install --index-url "$TORCH_INDEX" torch

echo ""
echo "[4/4] Installing zapbench (editable, --no-deps to avoid clobbering above)..."
pip install --no-deps -e "$ZAPBENCH_SRC"
# A couple of zapbench deps are noisy but we already have them; install the rest:
pip install absl-py clu connectomics distrax dm_pix gin-config grain immutabledict ml-collections tensorflow-probability tf-keras altair || true

echo ""
echo "============================================"
echo "  Venv built. Now stage cached data:"
echo "    source $VENV_DIR/bin/activate"
echo "    python $SCRIPT_DIR/prepare_data.py"
echo ""
echo "  ZAPBench data lives at gs://zapbench-release/ (public). The first"
echo "  prepare_data run streams + caches train+val arrays under"
echo "  $SCRIPT_DIR/zapbench_cache/ (gitignored, ~3-4 GB on disk)."
echo "============================================"
