#!/bin/bash
# =============================================================================
# scGPT Fine-tuning Script - CPU Optimized
# 
# Hardware: NVIDIA H20 (96GB HBM3) - CUDA torch not available in this env
# Uses CPU with optimized model size for reasonable training time
# PBMC 3K dataset (download-free)
# Expected: ARI 0.50-0.65 from scratch (no pretrained model available)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Project root directory (ABSOLUTE PATH)
# =============================================================================
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/8106b845-6b08-4773-a6f3-d059f983c960/scGPT/code/190002a6-8aa0-4d3b-b747-d5fdb361ade8/scGPT"

cd "${PROJECT_ROOT}"

# =============================================================================
# Python environment
# =============================================================================
PYTHON="python3"

# =============================================================================
# Experiment identification
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="scGPT_finetune_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/scgpt/run_log"
mkdir -p "${LOG_DIR}"

# Main log file
STDOUT_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "============================================" | tee -a "${STDOUT_LOG}"
echo "scGPT Fine-tuning Experiment" | tee -a "${STDOUT_LOG}"
echo "Experiment: ${EXPERIMENT_NAME}" | tee -a "${STDOUT_LOG}"
echo "Timestamp: $(date)" | tee -a "${STDOUT_LOG}"
echo "Project Root: ${PROJECT_ROOT}" | tee -a "${STDOUT_LOG}"
echo "============================================" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 0: Check environment
# =============================================================================
echo "[Step 0] Checking environment..." | tee -a "${STDOUT_LOG}"

# Check python
${PYTHON} --version 2>&1 | tee -a "${STDOUT_LOG}"

# Check GPU (for informational purposes)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | tee -a "${STDOUT_LOG}"
    echo "GPU available but training will run on CPU (CUDA torch not available)" | tee -a "${STDOUT_LOG}"
fi

# Check torch
${PYTHON} -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "${STDOUT_LOG}"

echo "Environment check complete." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1: Install dependencies (if needed)
# =============================================================================
echo "[Step 1] Checking and installing dependencies..." | tee -a "${STDOUT_LOG}"

# Ensure scGPT is installed in editable mode
cd "${PROJECT_ROOT}"
${PYTHON} -m pip install -e "${PROJECT_ROOT}" --no-deps --no-build-isolation 2>&1 | tee -a "${STDOUT_LOG}" | tail -5 || {
    echo "WARNING: pip install -e failed, continuing..." | tee -a "${STDOUT_LOG}"
}

# Install test dependencies
${PYTHON} -m pip install scib>=1.0.3 leidenalg>=0.8.10 2>&1 | tee -a "${STDOUT_LOG}" | tail -5 || echo "Some optional deps not installed" | tee -a "${STDOUT_LOG}"

echo "Dependencies ready." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 2: Check for pretrained model
# =============================================================================
echo "[Step 2] Checking for pretrained model..." | tee -a "${STDOUT_LOG}"

PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
if [ -f "${PRETRAINED_DIR}/best_model.pt" ]; then
    echo "Pretrained model found at ${PRETRAINED_DIR}/best_model.pt" | tee -a "${STDOUT_LOG}"
else
    echo "No pretrained model found. Training from scratch." | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Step 3: Run training
# =============================================================================
echo "[Step 3] Starting training..." | tee -a "${STDOUT_LOG}"

cd "${PROJECT_ROOT}/examples"

# Run with unbuffered output for real-time logging
# Use WANDB_MODE=dryrun to disable wandb cloud sync
# No CUDA_VISIBLE_DEVICES needed since torch is CPU-only
WANDB_MODE=dryrun \
${PYTHON} -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
    2>&1 | tee -a "${STDOUT_LOG}"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished with exit code ${TRAINING_EXIT_CODE}" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 4: Verify and summarize results
# =============================================================================
echo "[Step 4] Checking results..." | tee -a "${STDOUT_LOG}"

# Find latest save directory
LATEST_SAVE_DIR=$(ls -td "${PROJECT_ROOT}/examples/save/dev_PBMC_10K-"* 2>/dev/null | head -1)

if [ -n "${LATEST_SAVE_DIR}" ]; then
    echo "Latest save directory: ${LATEST_SAVE_DIR}" | tee -a "${STDOUT_LOG}"

    # Check for saved model
    if [ -f "${LATEST_SAVE_DIR}/best_model.pt" ]; then
        MODEL_SIZE=$(du -h "${LATEST_SAVE_DIR}/best_model.pt" | cut -f1)
        echo "Best model saved: ${LATEST_SAVE_DIR}/best_model.pt (${MODEL_SIZE})" | tee -a "${STDOUT_LOG}"
    fi

    # Check evaluation figures
    for f in "${LATEST_SAVE_DIR}"/embeddings_*.png; do
        if [ -f "$f" ]; then
            echo "  Evaluation figure: $(basename $f)" | tee -a "${STDOUT_LOG}"
        fi
    done

    # Check metrics summary
    if [ -f "${LATEST_SAVE_DIR}/metrics_summary.json" ]; then
        echo "Metrics summary:" | tee -a "${STDOUT_LOG}"
        ${PYTHON} -c "
import json
with open('${LATEST_SAVE_DIR}/metrics_summary.json') as f:
    d = json.load(f)
for k, v in d.items():
    if k != 'config':
        print(f'  {k}: {v}')
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi

    # Print final ARI from run.log
    if [ -f "${LATEST_SAVE_DIR}/run.log" ]; then
        echo "Best ARI from training log:" | tee -a "${STDOUT_LOG}"
        grep -o "best ARI: [0-9.]*" "${LATEST_SAVE_DIR}/run.log" | tail -1 | tee -a "${STDOUT_LOG}" || echo "  (not found in log)" | tee -a "${STDOUT_LOG}"
        tail -30 "${LATEST_SAVE_DIR}/run.log" | tee -a "${STDOUT_LOG}"
    fi
else
    echo "No save directory found. Check training output for errors." | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Summary
# =============================================================================
echo "============================================" | tee -a "${STDOUT_LOG}"
echo "Experiment Complete: ${EXPERIMENT_NAME}" | tee -a "${STDOUT_LOG}"
echo "Standard Output: ${STDOUT_LOG}" | tee -a "${STDOUT_LOG}"
echo "Training Exit Code: ${TRAINING_EXIT_CODE}" | tee -a "${STDOUT_LOG}"
echo "============================================" | tee -a "${STDOUT_LOG}"

exit ${TRAINING_EXIT_CODE}