#!/bin/bash
# Setup script for datasets/znaa (Zoned Neutral-Atom Architecture compilation)
# Creates a task-local venv that SimpleTES auto-detects at
# datasets/znaa/venv/ when running the znaa/znaa subtask.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -d "venv" ]]; then
    echo "Creating Python venv..."
    python3 -m venv venv
else
    echo "venv already exists, skipping..."
fi

echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate

echo ""
echo "=== Setup complete ==="
echo "SimpleTES will auto-detect datasets/znaa/venv/ when running znaa/znaa."
