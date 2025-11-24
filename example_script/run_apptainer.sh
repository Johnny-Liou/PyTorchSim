#!/bin/bash

# PyTorchSim Apptainer Launch Script
# This script sets up the environment and launches the Apptainer container

# Set Apptainer environment variables
# For example:
# export APPTAINER_CACHEDIR=/scratch/network/zl3193/images
# export APPTAINER_TMPDIR=/scratch/network/zl3193/images/tmp
export APPTAINER_CACHEDIR=
export APPTAINER_TMPDIR=

# Path to the Apptainer image
# APPTAINER_IMAGE="/scratch/network/zl3193/images/torchsim-ci_v1.0.0.sif"
APPTAINER_IMAGE=""

# Path to PyTorchSim directory (this script's directory)
PYTORCHSIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the image exists
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found at $APPTAINER_IMAGE"
    echo "Please make sure the image is downloaded."
    exit 1
fi

# Check if tmp directory exists
if [ ! -d "$APPTAINER_TMPDIR" ]; then
    echo "Creating temporary directory: $APPTAINER_TMPDIR"
    mkdir -p "$APPTAINER_TMPDIR"
fi

echo "=========================================="
echo "PyTorchSim Apptainer Container"
echo "=========================================="

apptainer shell \
  --nv \
  --writable-tmpfs \
  --no-home \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  --env TORCHSIM_DIR=/workspace/PyTorchSim \
  --env TORCHSIM_DUMP_PATH=/workspace/PyTorchSim-dev/output \
  --pwd /workspace/PyTorchSim-dev \
  --bind "$PYTORCHSIM_DIR:/workspace/PyTorchSim-dev" \
  "$APPTAINER_IMAGE"
