#!/bin/bash
# =============================================================================
# scGPT Fine-tuning with Prototype-based Contrastive Learning
# Cell Type-Aware Fine-tuning Script
#
# Hardware: NVIDIA H20 (96GB HBM3), CUDA 12.8
# Goal: Improve ARI from 0.71 to 0.78-0.82, avg_bio > 0.75
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Project root directory (ABSOLUTE PATH)
# =============================================================================
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/8106b845-6b08-4773-a6f3-d059f983c960/scGPT/code/190002a6-8aa0-4d3b-b747-d5fdb361ade8/scGPT"

cd "${PROJECT_ROOT}"

# =============================================================================
# Python environment configuration
# =============================================================================
# The system python3 path
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
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 2>/dev/null || echo "1")
echo "Found ${NUM_GPUS} GPU(s)" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1: Install dependencies into a virtual environment
# =============================================================================
echo "[Step 1] Setting up Python virtual environment..." | tee -a "${STDOUT_LOG}"

VENV_DIR="${PROJECT_ROOT}/.venv_scgpt"

# Check if we need to create a new venv (skip if already exists)
if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..." | tee -a "${STDOUT_LOG}"
    
    # Try to create a virtual environment (use --clear if exists but incomplete)
    rm -rf "${VENV_DIR}"
    ${PYTHON} -m venv "${VENV_DIR}" 2>&1 | tee -a "${STDOUT_LOG}"
    
    echo "Virtual environment created." | tee -a "${STDOUT_LOG}"
else
    echo "Virtual environment already exists at ${VENV_DIR}" | tee -a "${STDOUT_LOG}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..." | tee -a "${STDOUT_LOG}"
pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1b: Install PyTorch with CUDA support
# =============================================================================
echo "[Step 1b] Installing PyTorch with CUDA..." | tee -a "${STDOUT_LOG}"

# Determine CUDA version for PyTorch install
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "12.8")
echo "CUDA Driver version: ${CUDA_VERSION}" | tee -a "${STDOUT_LOG}"

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.8)
# Use the stable PyTorch 2.1.0 which is well-tested with scGPT
pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    2>&1 | tee -a "${STDOUT_LOG}"

echo "PyTorch installation completed." | tee -a "${STDOUT_LOG}"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1c: Install core dependencies (excluding scvi-tools to avoid version conflicts)
# =============================================================================
echo "[Step 1c] Installing core dependencies..." | tee -a "${STDOUT_LOG}"

# Install core scientific packages
pip install \
    pandas>=1.3.5 \
    scanpy>=1.9.1 \
    anndata>=0.8.0 \
    scikit-learn>=1.0.0 \
    matplotlib>=3.5.0 \
    tqdm>=4.64.0 \
    numpy>=1.22.0 \
    scipy>=1.7.0 \
    umap-learn>=0.5.3 \
    leidenalg>=0.8.10 \
    numba>=0.55.1 \
    2>&1 | tee -a "${STDOUT_LOG}"

echo "Core dependencies installed." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1d: Install scGPT from source (editable mode, with compatible deps)
# =============================================================================
echo "[Step 1d] Installing scGPT from source..." | tee -a "${STDOUT_LOG}"

# First install the package without dependencies (we already have them)
cd "${PROJECT_ROOT}"
pip install -e . --no-deps --no-build-isolation 2>&1 | tee -a "${STDOUT_LOG}"

echo "scGPT installation complete." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 2: Prepare pretrained model checkpoint
# =============================================================================
echo "[Step 2] Checking pretrained model..." | tee -a "${STDOUT_LOG}"

PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
mkdir -p "${PRETRAINED_DIR}"

if [ -f "${PRETRAINED_DIR}/best_model.pt" ]; then
    echo "Pretrained model found at ${PRETRAINED_DIR}" | tee -a "${STDOUT_LOG}"
    ls -la "${PRETRAINED_DIR}/" | tee -a "${STDOUT_LOG}"
else
    echo "No pretrained model found. Training from scratch." | tee -a "${STDOUT_LOG}"
    echo "For better results, download a pretrained checkpoint to:" | tee -a "${STDOUT_LOG}"
    echo "  ${PRETRAINED_DIR}/" | tee -a "${STDOUT_LOG}"
fi

# =============================================================================
# Step 3: Run prototype-based contrastive learning fine-tuning
# =============================================================================
echo "[Step 3] Starting prototype-based contrastive learning fine-tuning..." | tee -a "${STDOUT_LOG}"

cd "${PROJECT_ROOT}/examples"

# Run the finetune_integration.py with prototype contrastive learning
# 
# Key improvements enabled:
# - CLS head: Cell type classification with curriculum learning
# - Prototype contrastive head: EMA-updated prototypes with contrastive loss
# - CCE: Contrastive cell embedding objective
# - ECS: Elastic cell similarity regularization
# - DAR: Domain adversarial regularization for batch correction
# - Curriculum learning: cosine ramp-up for CLS and Proto weights
# - Large batch size: 128-256 for H20 96GB
# - AMP: Automatic mixed precision for faster training
#
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
python -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
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
        BEST_MODEL_SIZE=$(stat --format=%s "${LATEST_SAVE_DIR}/best_model.pt" 2>/dev/null)
        echo "Best model saved at: ${LATEST_SAVE_DIR}/best_model.pt (${BEST_MODEL_SIZE} bytes)" | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for ARI-best model
    if [ -f "${LATEST_SAVE_DIR}/best_model_ari.pt" ]; then
        echo "ARI-best model saved at: ${LATEST_SAVE_DIR}/best_model_ari.pt" | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for metrics summary
    if [ -f "${LATEST_SAVE_DIR}/metrics_summary.json" ]; then
        echo "Metrics summary:" | tee -a "${STDOUT_LOG}"
        python -c "
import json
with open('${LATEST_SAVE_DIR}/metrics_summary.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for scIB metrics
    if [ -f "${LATEST_SAVE_DIR}/scib_metrics.json" ]; then
        echo "scIB metrics:" | tee -a "${STDOUT_LOG}"
        python -c "
import json
with open('${LATEST_SAVE_DIR}/scib_metrics.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for evaluation UMAPs
    if ls "${LATEST_SAVE_DIR}"/embeddings_*.png 1>/dev/null 2>&1; then
        echo "Evaluation figures:" | tee -a "${STDOUT_LOG}"
        ls -la "${LATEST_SAVE_DIR}"/embeddings_*.png | tee -a "${STDOUT_LOG}"
    fi
    
    # Check run log
    if [ -f "${LATEST_SAVE_DIR}/run.log" ]; then
        echo "Last 100 lines of training log:" | tee -a "${STDOUT_LOG}"
        tail -100 "${LATEST_SAVE_DIR}/run.log" | tee -a "${STDOUT_LOG}"
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

# Deactivate virtual environment
if [ -n "${VIRTUAL_ENV}" ]; then
    deactivate 2>/dev/null || true
fi

exit ${TRAINING_EXIT_CODE}