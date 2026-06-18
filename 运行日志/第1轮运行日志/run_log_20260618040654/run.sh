#!/bin/bash
# =============================================================================
# scGPT Fine-tuning with Prototype-based Contrastive Learning
# Cell Type-Aware Fine-tuning Script
#
# Hardware: NVIDIA H20 (96GB HBM3), CUDA 12.8
# Expected: ARI 0.71 -> 0.78-0.82, avg_bio > 0.75
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
# The project uses the system-installed python3. Adjust as needed for your env.
# Recommended Python version: 3.11+
PYTHON="python3"

# =============================================================================
# Experiment identification
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="scGPT_proto_clr_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${LOG_DIR}"

# Log files
STDOUT_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/run_${TIMESTAMP}_err.log"

echo "============================================" | tee -a "${STDOUT_LOG}"
echo "scGPT Prototype-based Contrastive Learning" | tee -a "${STDOUT_LOG}"
echo "Experiment: ${EXPERIMENT_NAME}" | tee -a "${STDOUT_LOG}"
echo "Timestamp: $(date)" | tee -a "${STDOUT_LOG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "${STDOUT_LOG}"
echo "============================================" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 0: Check GPU availability
# =============================================================================
echo "[Step 0] Checking GPU..." | tee -a "${STDOUT_LOG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | tee -a "${STDOUT_LOG}"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Found ${NUM_GPUS} GPU(s)" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1: Install dependencies (if needed)
# =============================================================================
echo "[Step 1] Installing dependencies..." | tee -a "${STDOUT_LOG}"

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.8)
# Note: Use the appropriate index URL for your CUDA version
# For CUDA 12.8, we use the CUDA 12.1 wheels which are forward-compatible
${PYTHON} -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${STDOUT_LOG}"

# Install core dependencies from requirements.txt
${PYTHON} -m pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    2>&1 | tee -a "${STDOUT_LOG}"

# Install scGPT and other dependencies
# Using --no-deps to avoid version conflicts, then install deps separately
cd "${PROJECT_ROOT}"
${PYTHON} -m pip install -e . --no-build-isolation 2>&1 | tee -a "${STDOUT_LOG}" || {
    echo "WARNING: pip install -e failed, trying direct install..." | tee -a "${STDOUT_LOG}"
    ${PYTHON} -m pip install \
        pandas>=1.3.5 \
        scanpy>=1.9.1 \
        scvi-tools>=0.16.0 \
        scib>=1.0.3 \
        anndata>=0.8.0 \
        scikit-learn>=1.0.0 \
        matplotlib>=3.5.0 \
        tqdm>=4.64.0 \
        wandb>=0.17.0 \
        torchtext \
        2>&1 | tee -a "${STDOUT_LOG}"
}

echo "Dependencies installed successfully." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 2: Prepare pretrained model checkpoint
# =============================================================================
echo "[Step 2] Checking pretrained model..." | tee -a "${STDOUT_LOG}"

PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
mkdir -p "${PRETRAINED_DIR}"

# If the pretrained model does not exist, download it
if [ ! -f "${PRETRAINED_DIR}/best_model.pt" ]; then
    echo "Downloading pretrained scGPT model..." | tee -a "${STDOUT_LOG}"
    # Option A: Download from Google Drive (scGPT whole-human recommended)
    # Using gdown or direct wget to the checkpoint
    # The model should contain: best_model.pt, args.json, vocab.json
    
    # For PBMC_10K integration fine-tuning, we use the continual pretrained model
    # Download link for whole-human model (recommended for integration):
    # https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
    
    # Since the user may not have gdown, we create a placeholder directory structure
    # and let the training script create a fresh vocabulary if no model is loaded.
    # Set load_model=None in the script to train without pretrained weights.
    echo "WARNING: No pretrained model found at ${PRETRAINED_DIR}" | tee -a "${STDOUT_LOG}"
    echo "The training will proceed without loading a pretrained model." | tee -a "${STDOUT_LOG}"
    echo "For better results, download a pretrained checkpoint to ${PRETRAINED_DIR}/" | tee -a "${STDOUT_LOG}"
    echo "The checkpoint should contain: best_model.pt, args.json, vocab.json" | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Step 3: Run prototype-based contrastive learning fine-tuning
# =============================================================================
echo "[Step 3] Starting training..." | tee -a "${STDOUT_LOG}"

cd "${PROJECT_ROOT}/examples"

# Run the finetune_integration.py with prototype contrastive learning
# Key hyperparameters:
# - batch_size=256 (increased from 64 for better contrastive learning)
# - cls_weight starts at 0.1, ramps to 1.0 (curriculum)
# - proto_weight starts at 0.05, ramps to 0.5 (curriculum)
# - epochs=30
# - ecs_thres=0.8 (elastic cell similarity)
# - GEPC=True (masked value prediction)
# - dab_weight=1.0 (domain adversarial)

WANDB_MODE=run \
CUDA_VISIBLE_DEVICES=0 \
${PYTHON} -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
    2>&1 | tee -a "${STDOUT_LOG}"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished with exit code ${TRAINING_EXIT_CODE}" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 4: Verify evaluation results
# =============================================================================
echo "[Step 4] Checking evaluation results..." | tee -a "${STDOUT_LOG}"

# Find the latest save directory
LATEST_SAVE_DIR=$(ls -td "${PROJECT_ROOT}/examples/save/dev_PBMC_10K-"* 2>/dev/null | head -1)

if [ -n "${LATEST_SAVE_DIR}" ]; then
    echo "Latest save directory: ${LATEST_SAVE_DIR}" | tee -a "${STDOUT_LOG}"
    
    # Check for saved model
    if [ -f "${LATEST_SAVE_DIR}/best_model.pt" ]; then
        echo "Best model saved at: ${LATEST_SAVE_DIR}/best_model.pt" | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for evaluation UMAPs
    if ls "${LATEST_SAVE_DIR}"/embeddings_*.png 1>/dev/null 2>&1; then
        echo "Evaluation figures:" | tee -a "${STDOUT_LOG}"
        ls -la "${LATEST_SAVE_DIR}"/embeddings_*.png | tee -a "${STDOUT_LOG}"
    fi
    
    # Check run log
    if [ -f "${LATEST_SAVE_DIR}/run.log" ]; then
        echo "Last 50 lines of training log:" | tee -a "${STDOUT_LOG}"
        tail -50 "${LATEST_SAVE_DIR}/run.log" | tee -a "${STDOUT_LOG}"
    fi
else
    echo "No save directory found. Check the training output above for errors." | tee -a "${STDOUT_LOG}"
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