#!/bin/bash

# Scaling Experiments Script
# Runs batched GEMM tests with varying SA counts and matrix sizes
# Usage: ./run_scaling_experiments.sh

echo "=========================================="
echo "Starting SA Scaling Experiments"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Configuration arrays
SA_COUNTS=(2 4 6 8 12 16 24 32)
MATRIX_SIZES=(128 256 512 1024 2048 4096)
BATCH_SIZE=1

# Base directory for PyTorchSim
# SCRIPT_DIR="/home/zl3193/courses/ece580/PyTorchSim"
SCRIPT_DIR=""
cd "$SCRIPT_DIR"

# Count total experiments
TOTAL_EXPERIMENTS=$((${#SA_COUNTS[@]} * ${#MATRIX_SIZES[@]}))
CURRENT_EXPERIMENT=0

echo "Configuration:"
echo "  SA Counts: ${SA_COUNTS[*]}"
echo "  Matrix Sizes: ${MATRIX_SIZES[*]}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Total Experiments: ${TOTAL_EXPERIMENTS}"
echo ""
echo "=========================================="
echo ""

# Run experiments
for SIZE in "${MATRIX_SIZES[@]}"; do
    for SA in "${SA_COUNTS[@]}"; do
        CONFIG_FILE="systolic_ws_128x128_c2_sa${SA}_booksim_tpuv4.json"
        
        echo "--------------------------------------"
        echo "Testing with ${SA} Systolic Arrays"
        echo "Config: ${CONFIG_FILE}"
        echo "--------------------------------------"
        echo ""
    
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        MATRIX_SIZE="${BATCH_SIZE}x${SIZE}x${SIZE}x${SIZE}"
        
        echo "[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Running: SA=${SA}, Size=${SIZE}x${SIZE}x${SIZE}"
        echo "Matrix: ${MATRIX_SIZE}, Config: ${CONFIG_FILE}"
        echo ""
        
        # Run the experiment
        ./run_test_bGemm.sh "${MATRIX_SIZE}" "${CONFIG_FILE}"
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "✓ Experiment completed successfully"
        else
            echo "✗ Experiment failed (exit code: $?)"
        fi
        
        echo ""
        echo "=========================================="
        echo ""
    done
done

echo ""
echo "=========================================="
echo "All Scaling Experiments Complete!"
echo "Date: $(date)"
echo "Total Experiments Run: ${TOTAL_EXPERIMENTS}"
echo "=========================================="
