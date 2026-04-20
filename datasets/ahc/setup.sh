#!/bin/bash
# AHC Setup Script for Internal Machines
#
# Prerequisites:
#   - ale-bench.tar (Docker image) in this directory
#   - cache/ directory with test inputs (or cache_ahc.zip to extract)
#
# Usage:
#   cd datasets/ahc && bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================"
echo "AHC Setup"
echo "========================================"

# 1. Load Docker image
echo ""
echo "[1/5] Loading Docker image..."
if docker images | grep -q "yimjk/ale-bench.*cpp20-202301"; then
    echo "  Docker image already loaded."
else
    if [[ -f "ale-bench.tar" ]]; then
        docker load -i ale-bench.tar
        echo "  Docker image loaded."
    else
        echo "  ERROR: ale-bench.tar not found!"
        echo "  Please copy it to: ${SCRIPT_DIR}/ale-bench.tar"
        exit 1
    fi
fi

# 2. Extract cache if needed
echo ""
echo "[2/5] Checking cache..."
if [[ -d "cache/public_inputs_150" && -d "cache/tester_binaries" ]]; then
    echo "  Cache already exists."
else
    if [[ -f "cache_ahc.zip" ]]; then
        echo "  Extracting cache_ahc.zip..."
        unzip -o cache_ahc.zip -d cache/
        echo "  Cache extracted."
    elif [[ -d "cache" ]]; then
        echo "  Cache directory exists but may be incomplete."
    else
        echo "  ERROR: Neither cache/ nor cache_ahc.zip found!"
        echo "  Run 'bash get_cache.sh' on a machine with internet access,"
        echo "  then copy cache/ directory here."
        exit 1
    fi
fi

# 3. Make tester binaries executable
echo ""
echo "[3/5] Setting permissions on tester binaries..."
chmod +x cache/tester_binaries/* 2>/dev/null || true
echo "  Done."

# 4. Verify Docker
echo ""
echo "[4/5] Verifying Docker..."
DOCKER_OUTPUT=$(docker run --rm yimjk/ale-bench:cpp20-202301 g++-12 --version 2>&1 | head -1)
if [[ "${DOCKER_OUTPUT}" == *"g++-12"* ]]; then
    echo "  Docker OK: ${DOCKER_OUTPUT}"
else
    echo "  ERROR: Docker test failed!"
    echo "  Output: ${DOCKER_OUTPUT}"
    exit 1
fi

# 5. Test evaluator
echo ""
echo "[5/5] Testing evaluator (this takes ~60-90s)..."
cd "${SCRIPT_DIR}/../.."
export AHC_CACHE_DIR="${SCRIPT_DIR}/cache"
export AHC_CASE_WORKERS=4

EVAL_OUTPUT=$(python3 -c "
import sys
sys.path.insert(0, 'datasets/ahc/ahc039')
from evaluator import evaluate
result = evaluate('datasets/ahc/ahc039/init_program.py')
print(f\"Score: {result['combined_score']:.2f}, AC: {result['num_accepted']}/{result['num_cases']}\")
" 2>&1)

echo "  ${EVAL_OUTPUT}"

if [[ "${EVAL_OUTPUT}" == *"Score:"* && "${EVAL_OUTPUT}" == *"150/150"* ]]; then
    echo ""
    echo "========================================"
    echo "Setup complete!"
    echo "========================================"
    echo ""
    echo "To run SimpleTES on AHC:"
    echo "  python main_wizard.py          # interactive, pick ahc/ahc039 or ahc/ahc058"
else
    echo ""
    echo "WARNING: Evaluator test may have issues."
    echo "Check the output above."
fi
