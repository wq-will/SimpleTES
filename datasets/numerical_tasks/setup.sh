#!/bin/bash
# Setup script for numerical_tasks
# Run this once per machine to set up the task-level environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Setting up numerical_tasks environment ==="

# 1. Create venv if not exists
if [[ ! -d "venv" ]]; then
    echo "Creating Python venv..."
    python3 -m venv venv
else
    echo "venv already exists, skipping..."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate
echo "Python dependencies installed."

# 3. Download Eigen for lasso_path (C++ tasks)
if [[ -d "lasso_path" && ! -d "lasso_path/eigen" ]]; then
    echo "Downloading Eigen 3.4.0 for lasso_path..."
    curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen.tar.gz
    tar -xzf eigen.tar.gz
    mv eigen-3.4.0 lasso_path/eigen
    rm eigen.tar.gz
    echo "Eigen downloaded."
else
    echo "Eigen already exists or lasso_path not found, skipping..."
fi

# 4. Verify g++ is available (for C++ tasks)
if ! command -v g++ &> /dev/null; then
    echo "WARNING: g++ not found. Please install g++ for C++ tasks (e.g., apt install g++)"
else
    echo "g++ found: $(g++ --version | head -1)"
fi

echo ""
echo "=== Setup complete ==="
