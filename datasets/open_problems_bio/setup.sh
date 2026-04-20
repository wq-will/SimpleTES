#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Setup script for OpenProblems benchmark environment
#
# Prefers uv (if available), falls back to python -m venv + pip.
# Requires Python >= 3.11.
#
# Usage:
#   cd datasets/open_problems_bio
#   bash setup_openproblems.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

PYTHON_SPEC="${PYTHON_BIN:-}"

if [ -z "${PIP_INDEX_URL:-}" ]; then
  PIP_INDEX_URL="https://pypi.org/simple"
  export PIP_INDEX_URL
  echo "PIP_INDEX_URL not set; defaulting to PyPI: $PIP_INDEX_URL"
else
  echo "PIP_INDEX_URL: $PIP_INDEX_URL"
fi

echo "============================================"
echo "  Setting up OpenProblems benchmark environment"
echo "============================================"
echo "Venv:   $VENV_DIR"
echo ""

# ---------- helper: verify Python >= 3.11 inside venv ----------
verify_python() {
  "$VENV_DIR/bin/python" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):
    raise SystemExit(
        f"ERROR: Need Python >=3.11 in venv, got {sys.version.split()[0]}.\n"
        "Set PYTHON_BIN to a Python >=3.11 interpreter or ensure uv can install one."
    )
print("Python OK:", sys.version.split()[0])
PY
}

# ==========================================================
# 1. Create venv and install core dependencies
# ==========================================================
echo "[1/6] Creating venv and installing dependencies..."

if command -v uv >/dev/null 2>&1; then
  echo "Using uv for venv + installs (set UV_CACHE_DIR to override)."
  : "${UV_CACHE_DIR:=/tmp/uv-cache-openproblems-bio}"
  export UV_CACHE_DIR
  export UV_DEFAULT_INDEX="$PIP_INDEX_URL"

  if [ -z "$PYTHON_SPEC" ]; then
    PYTHON_SPEC=">=3.11"
  fi
  echo "Python: $PYTHON_SPEC"

  if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python "$PYTHON_SPEC" --prompt openproblems
  fi

  verify_python

  # Install pip/setuptools into the venv for convenience
  uv pip install --python "$VENV_DIR/bin/python" --upgrade pip setuptools wheel
  uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt

  # Activate so subsequent commands target this venv
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  USE_UV=true
else
  echo "uv not found; falling back to python -m venv (requires Python >=3.11 + venv/ensurepip)."

  if [ -z "$PYTHON_SPEC" ]; then
    PYTHON_SPEC="python3"
  fi
  if ! command -v "$PYTHON_SPEC" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON_SPEC' not found. Install Python >=3.11 or set PYTHON_BIN to it." >&2
    exit 1
  fi
  echo "Python: $PYTHON_SPEC"

  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_SPEC" -m venv "$VENV_DIR" --prompt openproblems || {
      echo "ERROR: failed to create venv with '$PYTHON_SPEC -m venv'." >&2
      echo "On Debian/Ubuntu you may need: apt install python3-venv" >&2
      exit 1
    }
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  verify_python

  python -m pip install --upgrade pip setuptools wheel requests
  python -m pip install -r requirements.txt

  USE_UV=false
fi

# ==========================================================
# 2. Install molecular-cross-validation (--no-deps)
# ==========================================================
echo ""
echo "[2/6] Installing molecular-cross-validation (no-deps)..."
if [ "$USE_UV" = true ]; then
  uv pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
else
  python -m pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
fi

# ==========================================================
# 3. Clone and checkout openproblems
# ==========================================================
echo ""
echo "[3/6] Cloning openproblems v1.0.0..."
if [ -d "openproblems" ]; then
    echo "  openproblems directory exists, skipping clone"
else
    git clone https://github.com/openproblems-bio/openproblems.git
fi
cd openproblems
git checkout v1.0.0 2>/dev/null || true

# ==========================================================
# 4. Apply patch
# ==========================================================
echo ""
echo "[4/6] Applying API fix patch..."
bash ../openproblems_api_fix.sh || echo "  Patch already applied or failed (check manually)"
cd ..

# ==========================================================
# 5. Install openproblems (--no-deps to avoid version conflicts)
# ==========================================================
echo ""
echo "[5/6] Installing openproblems (no-deps)..."
if [ "$USE_UV" = true ]; then
  uv pip install --no-deps -e ./openproblems
else
  python -m pip install --no-deps -e ./openproblems
fi

# ==========================================================
# 6. Ensure setuptools has pkg_resources (openproblems v1.0.0 needs it)
# ==========================================================
echo ""
echo "[6/6] Ensuring setuptools==75.8.2 for pkg_resources..."
if [ "$USE_UV" = true ]; then
  uv pip install "setuptools==75.8.2"
else
  python -m pip install "setuptools==75.8.2"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:"
echo "    source .venv/bin/activate"
echo ""
echo "  Or use with SimpleTES:"
echo "    python main.py ... --eval-venv datasets/open_problems_bio/.venv"
echo "============================================"