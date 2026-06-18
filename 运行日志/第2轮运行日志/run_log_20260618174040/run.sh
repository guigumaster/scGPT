#!/bin/bash
# =============================================================================
# scGPT Fine-tuning with Prototype-based Contrastive Learning
# Cell Type-Aware Fine-tuning Script
#
# Hardware: NVIDIA H20 (96GB HBM3), CUDA 12.8 (CPU PyTorch fallback)
# Goal: Improve ARI through prototype contrastive learning and curriculum training
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Project root directory (ABSOLUTE PATH)
# =============================================================================
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/8106b845-6b08-4773-a6f3-d059f983c960/scGPT/code/190002a6-8aa0-4d3b-b747-d5fdb361ade8/scGPT"

cd "${PROJECT_ROOT}"

# =============================================================================
# Python environment - Using /tmp venv for fast I/O
# =============================================================================
VENV_DIR="/tmp/scgpt_venv"
PYTHON="${VENV_DIR}/bin/python"

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
echo "============================================" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 0: Check GPU availability
# =============================================================================
echo "[Step 0] Checking GPU..." | tee -a "${STDOUT_LOG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 | tee -a "${STDOUT_LOG}" || echo "nvidia-smi not available"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 2>/dev/null || echo "0")
echo "Found ${NUM_GPUS} GPU(s)" | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1: Setup virtual environment (on /tmp for fast I/O)
# =============================================================================
echo "[Step 1] Setting up Python virtual environment on /tmp..." | tee -a "${STDOUT_LOG}"

if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..." | tee -a "${STDOUT_LOG}"
    rm -rf "${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" --without-pip 2>&1 | tee -a "${STDOUT_LOG}"
    
    # Install pip
    python3 -m pip install pip --target "${VENV_DIR}/lib/python3.14/site-packages/" --no-index --find-links /tmp/pip-unpack-cjyl__v7/ 2>&1 | tee -a "${STDOUT_LOG}"
    
    echo "Virtual environment created at ${VENV_DIR}." | tee -a "${STDOUT_LOG}"
else
    echo "Virtual environment already exists at ${VENV_DIR}" | tee -a "${STDOUT_LOG}"
fi

# Verify the venv
echo "Verifying virtual environment..." | tee -a "${STDOUT_LOG}"
${VENV_DIR}/bin/python --version 2>&1 | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1b: Install PyTorch with CPU support (CUDA not available for Python 3.14)
# =============================================================================
echo "[Step 1b] Installing PyTorch (CPU)..." | tee -a "${STDOUT_LOG}"

# Check if torch is already installed
if ${VENV_DIR}/bin/python -c "import torch; print('OK')" 2>/dev/null; then
    echo "PyTorch already installed." | tee -a "${STDOUT_LOG}"
else
    echo "Installing PyTorch CPU from cached wheel..." | tee -a "${STDOUT_LOG}"
    
    # Install core dependencies first
    ${VENV_DIR}/bin/python -m pip install typing_extensions --no-index --find-links /tmp/pip-unpack-wxfhm0db/ 2>&1 | tee -a "${STDOUT_LOG}"
    
    # Install torch CPU from pip cache
    ${VENV_DIR}/bin/python -m pip install torch --no-index --find-links /tmp/pip-unpack-conttyye/ --no-deps 2>&1 | tee -a "${STDOUT_LOG}"
    
    # Install remaining deps
    ${VENV_DIR}/bin/python -m pip install sympy filelock networkx jinja2 MarkupSafe fsspec mpmath setuptools --no-index --find-links /tmp/pip-unpack-h8zowxou/ 2>&1 | tee -a "${STDOUT_LOG}"
fi

echo "PyTorch installation completed." | tee -a "${STDOUT_LOG}"
${VENV_DIR}/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1c: Install core dependencies
# =============================================================================
echo "[Step 1c] Installing core dependencies..." | tee -a "${STDOUT_LOG}"

# Check if required packages are installed
if ${VENV_DIR}/bin/python -c "import scanpy; print('OK')" 2>/dev/null; then
    echo "Core dependencies already installed." | tee -a "${STDOUT_LOG}"
else
    echo "Installing core dependencies..." | tee -a "${STDOUT_LOG}"
    ${VENV_DIR}/bin/python -m pip install numpy pandas scipy scikit-learn matplotlib tqdm 2>&1 | tee -a "${STDOUT_LOG}"
    ${VENV_DIR}/bin/python -m pip install scanpy anndata leidenalg umap-learn 2>&1 | tee -a "${STDOUT_LOG}"
    ${VENV_DIR}/bin/python -m pip install datasets 2>&1 | tee -a "${STDOUT_LOG}"
fi

echo "Core dependencies installed." | tee -a "${STDOUT_LOG}"

# =============================================================================
# Step 1d: Install scGPT from source (editable mode, with compatible deps)
# =============================================================================
echo "[Step 1d] Installing scGPT from source..." | tee -a "${STDOUT_LOG}"

if ${VENV_DIR}/bin/python -c "import scgpt; print('OK')" 2>/dev/null; then
    echo "scGPT already installed." | tee -a "${STDOUT_LOG}"
else
    cd "${PROJECT_ROOT}"
    ${VENV_DIR}/bin/python -m pip install -e . --no-deps --no-build-isolation 2>&1 | tee -a "${STDOUT_LOG}"
fi

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
# - AMP: Automatic mixed precision (GPU only)
#
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
${VENV_DIR}/bin/python -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
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
        ${VENV_DIR}/bin/python -c "
import json
with open('${LATEST_SAVE_DIR}/metrics_summary.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
" 2>&1 | tee -a "${STDOUT_LOG}"
    fi
    
    # Check for scIB metrics
    if [ -f "${LATEST_SAVE_DIR}/scib_metrics.json" ]; then
        echo "scIB metrics:" | tee -a "${STDOUT_LOG}"
        ${VENV_DIR}/bin/python -c "
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
        echo "Last 50 lines of training log:" | tee -a "${STDOUT_LOG}"
        tail -50 "${LATEST_SAVE_DIR}/run.log" | tee -a "${STDOUT_LOG}"
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