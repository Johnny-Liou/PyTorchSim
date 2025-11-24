#!/bin/bash

# Helper script to run batched GEMM tests with structured naming
# Usage: ./run_test_bGemm.sh [MATRIX_SIZE] [CONFIG_FILE]
# Example: ./run_test_bGemm.sh 4x512x512x512 systolic_ws_128x128_c2_sa8_booksim_tpuv3.json
#
# MATRIX_SIZE format: BxMxKxN
#   B = batch size
#   M = rows of matrix A
#   K = shared dimension (cols of A, rows of B)
#   N = cols of matrix B

# Default values
DEFAULT_MATRIX_SIZE="1x512x512x512"
DEFAULT_CONFIG="systolic_ws_128x128_c2_sa4_booksim_tpuv4.json"

# Get matrix size from argument or use default
MATRIX_SIZE=${1:-$DEFAULT_MATRIX_SIZE}

# Get config file from argument or use default
CONFIG_FILE=${2:-$DEFAULT_CONFIG}

echo "=========================================="
echo "Running Batched GEMM Test"
echo "Matrix Size: ${MATRIX_SIZE}"
echo "Config: ${CONFIG_FILE}"
echo "=========================================="
echo ""

# Sync configs from dev to container's PyTorchSim directory
echo "Syncing configuration files..."
if [ -d "/workspace/PyTorchSim/PyTorchSimBackend/configs" ]; then
    echo "  Copying configs from PyTorchSim-dev to PyTorchSim..."
    cp -ru /workspace/PyTorchSim-dev/PyTorchSimBackend/configs/* \
           /workspace/PyTorchSim/PyTorchSimBackend/configs/ 2>/dev/null || true
    echo "  Config sync complete!"
else
    echo "  Warning: /workspace/PyTorchSim not found (running outside container?)"
fi
echo ""

# Export parameters for config_bGemm.sh to use
export BGEMM_MATRIX_SIZE="${MATRIX_SIZE}"
export BGEMM_CONFIG_FILE="${CONFIG_FILE}"

# Source the configuration script (it will handle all setup)
source /workspace/PyTorchSim-dev/config_bGemm.sh

# Parse matrix dimensions for python command
IFS='x' read -r BATCH M K N <<< "$MATRIX_SIZE"

echo ""
echo "Starting test execution..."
echo ""

# Run the batched GEMM test with parsed dimensions
python /workspace/PyTorchSim-dev/tests/test_bGemm.py --batch ${BATCH} --m ${M} --k ${K} --n ${N}

# Summary
echo ""
echo "=========================================="
echo "Test Complete!"
echo "Results saved to: ${TORCHSIM_DUMP_PATH}"
echo "=========================================="
