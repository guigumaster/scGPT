#!/bin/bash
# =============================================================================
# scGPT v4 Fine-tuning for scRNA-seq Integration
# Optimized for from-scratch training on PBMC 3K with NVIDIA H20 (96GB)
#
# Based on proven v2 architecture (ARI=0.5999) with targeted improvements:
# - Small model: 128-dim, 3-layer, 4-head (proven for small data)
# - Small batch: 32 (proven for from-scratch training)
# - All losses: MLM + MVC + CLS + Proto + CCE + DAB (proven)
# - More epochs: 150 with cosine LR schedule
# - Better initialization, stratified batches, early stopping
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Project root directory (ABSOLUTE PATH)
# =============================================================================
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/8106b845-6b08-4773-a6f3-d059f983c960/scGPT/code/190002a6-8aa0-4d3b-b747-d5fdb361ade8/scGPT"

cd "${PROJECT_ROOT}"

# =============================================================================
# Use the system conda environment with CUDA PyTorch
# =============================================================================
CONDA_PYTHON="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python3"

# =============================================================================
# Experiment identification
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="scGPT_v4_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${LOG_DIR}"

# Log files
STDOUT_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/run_${TIMESTAMP}_err.log"

echo "============================================" | tee -a "${STDOUT_LOG}"
echo "scGPT v4 Fine-tuning (from-scratch optimized)" | tee -a "${STDOUT_LOG}"
echo "Experiment: ${EXPERIMENT_NAME}" | tee -a "${STDOUT_LOG}"
echo "Timestamp: $(date)" | tee -a "${STDOUT_LOG}"
echo "============================================" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 0: Check GPU availability
# =============================================================================
echo "[Step 0] Checking GPU..." | tee -a "${STDOUT_LOG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 | tee -a "${STDOUT_LOG}" || echo "nvidia-smi not available"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 2>/dev/null || echo "0")
echo "Found ${NUM_GPUS} GPU(s)" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1: Verify the conda environment with CUDA PyTorch
# =============================================================================
echo "[Step 1] Verifying conda environment with CUDA PyTorch..." | tee -a "${STDOUT_LOG}"
${CONDA_PYTHON} --version 2>&1 | tee -a "${STDOUT_LOG}"
${CONDA_PYTHON} -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" 2>&1 | tee -a "${STDOUT_LOG}"

echo "Using Python: ${CONDA_PYTHON}" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 2: Check scGPT installation
# =============================================================================
echo "[Step 2] Checking scGPT installation..." | tee -a "${STDOUT_LOG}"
cd "${PROJECT_ROOT}"

# Install scGPT from source if not installed
if ${CONDA_PYTHON} -c "import scgpt; print('scGPT OK')" 2>/dev/null; then
    echo "scGPT already installed." | tee -a "${STDOUT_LOG}"
else
    echo "Installing scGPT from source..." | tee -a "${STDOUT_LOG}"
    ${CONDA_PYTHON} -m pip install -e . --no-deps --no-build-isolation 2>&1 | tee -a "${STDOUT_LOG}"
    echo "scGPT installation complete." | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Step 3: Prepare pretrained model checkpoint (optional)
# =============================================================================
echo "[Step 3] Checking pretrained model..." | tee -a "${STDOUT_LOG}"

PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
mkdir -p "${PRETRAINED_DIR}"

if [ -f "${PRETRAINED_DIR}/best_model.pt" ]; then
    echo "Pretrained model found at ${PRETRAINED_DIR}" | tee -a "${STDOUT_LOG}"
    ls -la "${PRETRAINED_DIR}/" | tee -a "${STDOUT_LOG}"
else
    echo "No pretrained model found. Training from scratch with Xavier init." | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Step 4: Run v4 fine-tuning
# =============================================================================
echo "[Step 4] Starting v4 fine-tuning..." | tee -a "${STDOUT_LOG}"

cd "${PROJECT_ROOT}/examples"

# Run the finetune_integration.py v4
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
${CONDA_PYTHON} -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
    2>&1 | tee -a "${STDOUT_LOG}"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished with exit code ${TRAINING_EXIT_CODE}" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 5: Verify evaluation results
# =============================================================================
echo "[Step 5] Checking evaluation results..." | tee -a "${STDOUT_LOG}"

# Find the latest save directory
LATEST_SAVE_DIR=$(ls -td "${PROJECT_ROOT}/examples/save/dev_PBMC_10K-"* 2>/dev/null | head -1)

if [ -n "${LATEST_SAVE_DIR}" ]; then
    echo "Latest save directory: ${LATEST_SAVE_DIR}" | tee -a "${STDOUT_LOG}"
    
    # Check for saved model
    if [ -f "${LATEST_SAVE_DIR}/best_model.pt" ]; then
        BEST_MODEL_SIZE=$(stat --format=%s "${LATEST_SAVE_DIR}/best_model.pt" 2>/dev/null || echo "unknown")
        echo "Best model saved at: ${LATEST_SAVE_DIR}/best_model.pt (${BEST_MODEL_SIZE} bytes)" | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for ARI-best model
    if [ -f "${LATEST_SAVE_DIR}/best_model_ari.pt" ]; then
        echo "ARI-best model saved at: ${LATEST_SAVE_DIR}/best_model_ari.pt" | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for metrics summary
    if [ -f "${LATEST_SAVE_DIR}/metrics_summary.json" ]; then
        echo "Metrics summary:" | tee -a "${STDOUT_LOG}"
        ${CONDA_PYTHON} -c "
import json
with open('${LATEST_SAVE_DIR}/metrics_summary.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for scIB metrics
    if [ -f "${LATEST_SAVE_DIR}/scib_metrics.json" ]; then
        echo "scIB metrics:" | tee -a "${STDOUT_LOG}"
        ${CONDA_PYTHON} -c "
import json
with open('${LATEST_SAVE_DIR}/scib_metrics.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi
    
    # Check run log
    if [ -f "${LATEST_SAVE_DIR}/run.log" ]; then
        echo "Last 30 lines of training log:" | tee -a "${STDOUT_LOG}"
        tail -30 "${LATEST_SAVE_DIR}/run.log" | tee -a "${STDOUT_LOG}"
    fi
else
    echo "No save directory found. Checking error log..." | tee -a "${STDOUT_LOG}"
    if [ -f "${STDERR_LOG}" ]; then
        echo "Last 30 lines of stderr:" | tee -a "${STDOUT_LOG}"
        tail -30 "${STDERR_LOG}" | tee -a "${STDOUT_LOG}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo "============================================" | tee -a "${STDOUT_LOG}"
echo "Experiment Complete: ${EXPERIMENT_NAME}" | tee -a "${STDOUT_LOG}"
echo "Project Root: ${PROJECT_ROOT}" | tee -a "${STDOUT_LOG}"
echo "Standard Output: ${STDOUT_LOG}" | tee -a "${STDOUT_LOG}"
echo "Standard Error: ${STDERR_LOG}" | tee -a "${STDOUT_LOG}"
echo "Training Exit Code: ${TRAINING_EXIT_CODE}" | tee -a "${STDOUT_LOG}"
echo "============================================" | tee -a "${STDOUT_LOG}"

exit ${TRAINING_EXIT_CODE}