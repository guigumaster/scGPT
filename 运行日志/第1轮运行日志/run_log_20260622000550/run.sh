#!/bin/bash
# Running script for scGPT Tutorial_Integration.py
# This script sets up the environment and runs the integration training

# Set the project root directory
export PROJECT_ROOT=/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/d784c0e3-1f65-4add-902a-ed5bc652c726/scGPT/code/511636f8-ea33-4131-9524-324875b4b630/scGPT

# Use the base conda Python (has working torch 2.11.0+cu128)
export PYTHON=/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python3

# Navigate to the project root
cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON"
echo "Python version: $($PYTHON --version)"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=disabled

# Auto-select the GPU with the most free memory
echo "Checking GPU status..."
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | xargs)
if [ -z "$GPU_ID" ]; then
    echo "Warning: No GPU found, using CPU mode"
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Using GPU $GPU_ID (most free memory)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv | head -2
fi

echo ""
echo "Starting training at $(date)"
echo "=========================================="

# Run the integration tutorial using absolute paths
$PYTHON -u "$PROJECT_ROOT/tutorials/Tutorial_Integration.py" 2>&1 | tee "$PROJECT_ROOT/run_log/training_output_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?
echo "=========================================="
echo "Training finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE