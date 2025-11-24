#!/bin/bash

# Configuration script for PyTorchSim batched GEMM testing
# Usage: source config_bGemm.sh (inside the apptainer container)
# Note: This script expects BGEMM_MATRIX_SIZE and BGEMM_CONFIG_FILE to be set

echo "=========================================="
echo "Setting PyTorchSim Configuration"
echo "=========================================="

# ===== Test Parameters =====
# Matrix size for batched GEMM (format: BxMxKxN)
# B = batch size, M = rows, K = shared dimension, N = columns
# Use exported value from run_test_bGemm.sh or default
MATRIX_SIZE="${BGEMM_MATRIX_SIZE:-1x512x512x512}"

# Parse matrix dimensions from MATRIX_SIZE
IFS='x' read -r BATCH M K N <<< "$MATRIX_SIZE"

echo "Batched GEMM Dimensions:"
echo "  Batch Size (B): ${BATCH}"
echo "  M (rows of A): ${M}"
echo "  K (shared dim): ${K}"
echo "  N (cols of B): ${N}"
echo "  Matrix A: [${BATCH}, ${M}, ${K}]"
echo "  Matrix B: [${BATCH}, ${K}, ${N}]"
echo "  Output:   [${BATCH}, ${M}, ${N}]"

# ===== Backend Configuration =====
# Use exported config file or default
CONFIG_FILE="${BGEMM_CONFIG_FILE:-systolic_ws_128x128_c2_sa8_booksim_tpuv3.json}"
export TORCHSIM_CONFIG="/workspace/PyTorchSim-dev/PyTorchSimBackend/configs/${CONFIG_FILE}"
echo ""
echo "Backend Config: $TORCHSIM_CONFIG"

# Extract config parameters from filename
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
SYSTOLIC_SIZE=$(echo "$CONFIG_NAME" | grep -oP 'ws_\K[0-9]+x[0-9]+')
NUM_CORES=$(echo "$CONFIG_NAME" | grep -oP '_c\K[0-9]+')
SA_PER_CORE=$(echo "$CONFIG_NAME" | grep -oP '_sa\K[0-9]+')
INTERCONNECT=$(echo "$CONFIG_NAME" | grep -oP 'sa[0-9]+_\K[^_]+_[^_]+' || echo "$CONFIG_NAME" | grep -oP 'c[0-9]+_\K[^_]+_[^_]+')

echo "  Systolic Array: ${SYSTOLIC_SIZE} per core"
echo "  Number of Cores: ${NUM_CORES}"
echo "  Systolic Arrays per Core: ${SA_PER_CORE:-1}"
echo "  Interconnect: ${INTERCONNECT}"

# ===== Tile Size Configuration =====
export TORCHSIM_MANUAL_TILE_SIZE=1

if [ "$TORCHSIM_MANUAL_TILE_SIZE" -eq 1 ]; then
    echo ""
    echo "Manual Tile Size: ENABLED"
    export TORCHSIM_TILE_M=128
    export TORCHSIM_TILE_N=128
    export TORCHSIM_TILE_K=128
    TILE_SIZE="${TORCHSIM_TILE_M}x${TORCHSIM_TILE_N}x${TORCHSIM_TILE_K}"
    echo "  Tile Size: M=${TORCHSIM_TILE_M}, N=${TORCHSIM_TILE_N}, K=${TORCHSIM_TILE_K}"
else
    echo ""
    echo "Manual Tile Size: DISABLED (using automatic tile selection)"
    TILE_SIZE="auto"
fi

# ===== Subtile Size Configuration =====
export TORCHSIM_SUBTILE=1
export TORCHSIM_MANUAL_SUBTILE_SIZE=1

if [ "$TORCHSIM_SUBTILE" -eq 1 ] && [ "$TORCHSIM_MANUAL_SUBTILE_SIZE" -eq 1 ]; then
    echo ""
    echo "Manual Subtile: ENABLED"
    export TORCHSIM_SUBTILE_M=128
    export TORCHSIM_SUBTILE_N=128
    export TORCHSIM_SUBTILE_K=128
    SUBTILE_SIZE="${TORCHSIM_SUBTILE_M}x${TORCHSIM_SUBTILE_N}x${TORCHSIM_SUBTILE_K}"
    echo "  Subtile Size: M=${TORCHSIM_SUBTILE_M}, N=${TORCHSIM_SUBTILE_N}, K=${TORCHSIM_SUBTILE_K}"
elif [ "$TORCHSIM_SUBTILE" -eq 1 ]; then
    echo ""
    echo "Subtile: ENABLED (automatic subtile selection)"
    SUBTILE_SIZE="auto"
else
    echo ""
    echo "Subtile: DISABLED"
    SUBTILE_SIZE="none"
fi

# ===== Output Directory Configuration =====
# Format: bGEMM/b{batch}_{MxKxN}_{cores}_{systolic}_{noc}_{tilesize}_{subtilesize}/
OUTPUT_DIR="bGEMM/b${BATCH}_${M}x${K}x${N}_c${NUM_CORES}_sa${SA_PER_CORE}_${SYSTOLIC_SIZE}_${INTERCONNECT}_tile${TILE_SIZE}_sub${SUBTILE_SIZE}"
export TORCHSIM_DUMP_PATH="/workspace/PyTorchSim-dev/output/${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Output Directory Structure:"
echo "  ${OUTPUT_DIR}"
echo ""
echo "Full Path: ${TORCHSIM_DUMP_PATH}"

echo "=========================================="
echo "Configuration Complete!"
echo "Results will be saved to:"
echo "  - Compilation: ${TORCHSIM_DUMP_PATH}/tmp/<hash>/"
echo "  - Simulation:  ${TORCHSIM_DUMP_PATH}/tmp/<hash>/backendsim_result/"
echo "  - Runtime:     ${TORCHSIM_DUMP_PATH}/tmp/<hash>/runtime_XXXX/"
echo ""
echo "You can now run: python tests/test_bGemm.py --batch ${BATCH} --m ${M} --k ${K} --n ${N}"
echo "=========================================="
