#!/bin/bash
# =============================================================================
# scGPT Enhanced Fine-tuning Pipeline (FIXED VERSION)
# =============================================================================
# This script runs the complete training pipeline:
#   1. Environment setup and dependency check
#   2. Pre-training data preparation
#   3. Enhanced fine-tuning with Norman continual pre-training + CCE
#   4. Final evaluation with scIB metrics (ARI, NMI, ASW, etc.)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Project root directory (ABSOLUTE PATH)
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/b88ba9d2-da3e-446d-a909-e27c3f575abd/scGPT/code/e129a197-dfc7-47f5-8816-32638fa1f8bd/scGPT"

# Python executable (absolute path to conda python)
PYTHON="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python3"

# Model checkpoint directory
MODEL_DIR="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/test_example/data/scgpt_integration/checkpoints/scGPT_human"

# Norman dataset path
NORMAN_DATA="${PROJECT_ROOT}/tutorials/data/norman.h5ad"

# Output directory for model saves and logs
SAVE_DIR="${PROJECT_ROOT}/tutorials/save"

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/run_log"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# WandB settings - use disabled mode to avoid API key requirement
# (metrics are logged to local files and visible in the training log)
export WANDB_MODE="disabled"
export WANDB_PROJECT="scGPT"
export WANDB_API_KEY="dummy-offline-key-00000000"

# GPU selection - auto-select GPU with most free memory
# Use nvidia-smi to find the GPU with the most free memory
__FREE_MEM=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | sort -t, -k2 -rn | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES="${__FREE_MEM:-0}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Suppress Python warnings for cleaner logs
export PYTHONWARNINGS="ignore"

# Fix setuptools compatibility with Python 3.13
# The old setuptools (59.5.0) has pkgutil.ImpImporter which was removed in Python 3.13
echo "Upgrading setuptools for Python 3.13 compatibility..."
${PYTHON} -m pip install --upgrade setuptools 2>&1 | tail -3

# Fix wandb compatibility (numpy 2.x compatibility)
echo "Installing compatible wandb version..."
${PYTHON} -m pip install "wandb>=0.17,<0.19" 2>&1 | tail -3

# Set OMP and MKL threads for performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

print_section() {
    log ""
    log "====================================================================="
    log "  $1"
    log "====================================================================="
}

# =============================================================================
# Step 0: Environment Setup
# =============================================================================

print_section "Step 0: Environment Setup"

cd "${PROJECT_ROOT}"
log "Working directory: $(pwd)"

# Make sure log directory exists
mkdir -p "${LOG_DIR}"
mkdir -p "${SAVE_DIR}"

# Check Python
log "Python: $(${PYTHON} --version 2>&1)"

# Check CUDA availability
${PYTHON} -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
" 2>&1 | tee -a "${LOG_FILE}"

# Check GPU memory
log "Checking GPU memory availability..."
${PYTHON} -c "
import subprocess
from io import StringIO
import pandas as pd

try:
    gpu_stats = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=index,memory.used,memory.free,name']).decode('utf-8')
    gpu_df = pd.read_csv(StringIO(gpu_stats), names=['index', 'memory.used', 'memory.free', 'name'], skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
    print('GPU availability:')
    print(gpu_df.to_string(index=False))
except Exception as e:
    print(f'Error checking GPU: {e}')
" 2>&1 | tee -a "${LOG_FILE}"

# Check dependencies
log "Checking Python dependencies..."
${PYTHON} -c "
import sys
missing = []
try:
    import torch; print(f'  torch {torch.__version__}')
except: missing.append('torch')
try:
    import scanpy; print(f'  scanpy {scanpy.__version__}')
except: missing.append('scanpy')
try:
    import wandb; print(f'  wandb {wandb.__version__}')
except: missing.append('wandb')
try:
    import scib; print(f'  scib')
except: missing.append('scib')
try:
    import anndata; print(f'  anndata {anndata.__version__}')
except: missing.append('anndata')

if missing:
    print(f'Missing packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
" 2>&1 | tee -a "${LOG_FILE}"

# Check model checkpoint
if [ ! -f "${MODEL_DIR}/best_model.pt" ]; then
    log "ERROR: Model checkpoint not found at ${MODEL_DIR}/best_model.pt"
    log "Please verify the path: ${MODEL_DIR}"
    ls -la "${MODEL_DIR}" 2>/dev/null || log "Directory does not exist!"
    exit 1
fi
log "Model checkpoint found: ${MODEL_DIR}/best_model.pt"

# Check Norman dataset
if [ ! -f "${NORMAN_DATA}" ]; then
    log "WARNING: Norman dataset not found at ${NORMAN_DATA}"
    log "The pipeline will skip Norman continual pre-training and proceed with direct fine-tuning."
else
    log "Norman dataset found: ${NORMAN_DATA} ($(du -h ${NORMAN_DATA} | cut -f1))"
fi

# =============================================================================
# Step 1: Data Preparation
# =============================================================================

print_section "Step 1: Data Preparation"

mkdir -p "${PROJECT_ROOT}/tutorials/data"

if [ -f "${NORMAN_DATA}" ]; then
    log "Norman dataset ready: ${NORMAN_DATA}"
fi

log "PBMC 10K will be downloaded automatically by the training script (first run only)."

# =============================================================================
# Step 2: Training - Enhanced Fine-tuning
# =============================================================================

print_section "Step 2: Enhanced Fine-tuning with Norman Pre-training + CCE"

log "Starting enhanced training pipeline..."
log "Key improvements enabled:"
if [ -f "${NORMAN_DATA}" ]; then
    log "  - Norman continual pre-training (${NORMAN_DATA})"
fi
log "  - Contrastive Cell Embedding (CCE) objective"
log "  - Adaptive mask ratio curriculum (0.55 -> 0.25)"
log "  - Cosine annealing LR scheduler with warm restarts"
log "  - Extended training (35 epochs)"
log "  - Early stopping on avg_bio"
log "  - Case-insensitive vocabulary matching for Norman data"

# Run the enhanced training script
cd "${PROJECT_ROOT}"

# WandB mode set above
# WANDB_MODE, WANDB_PROJECT, and WANDB_API_KEY are already exported

log "Running: ${PYTHON} -u ${PROJECT_ROOT}/tutorials/Tutorial_Integration.py"
log ""

# IMPORTANT: Use PIPESTATUS to capture python's exit code, not tee's
${PYTHON} -u "${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    log "ERROR: Training failed with exit code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi
log "Training completed successfully!"

# =============================================================================
# Step 3: Evaluation Summary
# =============================================================================

print_section "Step 3: Final Evaluation Summary"

log "Evaluation metrics are computed automatically during training."
log "Final metrics are logged to WandB project: ${WANDB_PROJECT}"
log "Checkpoints and UMAP plots are saved to: ${SAVE_DIR}"

# Find the latest save directory
LATEST_SAVE=$(ls -td "${SAVE_DIR}"/dev_* 2>/dev/null | head -1)
if [ -n "${LATEST_SAVE}" ]; then
    log "Latest save directory: ${LATEST_SAVE}"

    # Check for best model
    if [ -f "${LATEST_SAVE}/best_model.pt" ]; then
        MODEL_SIZE=$(du -h "${LATEST_SAVE}/best_model.pt" | cut -f1)
        log "Best model saved: ${LATEST_SAVE}/best_model.pt (${MODEL_SIZE})"
    fi

    # Check for evaluation images
    shopt -s nullglob
    for img in "${LATEST_SAVE}/embeddings_"*".png"; do
        log "  UMAP plot: $(basename ${img})"
    done

    # Check for Norman checkpoint
    if [ -f "${LATEST_SAVE}/norman_continual_pretrain.pt" ]; then
        NORMAN_CKPT_SIZE=$(du -h "${LATEST_SAVE}/norman_continual_pretrain.pt" | cut -f1)
        log "Norman pre-training checkpoint: ${LATEST_SAVE}/norman_continual_pretrain.pt (${NORMAN_CKPT_SIZE})"
    fi

    # Check for best model by avg_bio
    if [ -f "${LATEST_SAVE}/best_model_by_avg_bio.pt" ]; then
        log "Best model by avg_bio: ${LATEST_SAVE}/best_model_by_avg_bio.pt"
    fi
fi

# =============================================================================
# Step 4: Display Final Results Summary
# =============================================================================

print_section "Step 4: Results Summary"

log "Training pipeline completed."
log ""
log "=== Key Metrics to Track ==="
log "  avg_bio     - Average of NMI, ARI, and ASW_label (higher is better)"
log "  ARI         - Adjusted Rand Index for clustering (higher is better)"
log "  NMI         - Normalized Mutual Information (higher is better)"
log "  ASW_label   - Cell-type ASW (higher is better)"
log "  PCR_batch   - PCR batch effect removal (higher is better)"
log "  graph_conn  - Graph connectivity (higher is better)"
log ""
log "=== Output Files ==="
log "  Best model:           ${SAVE_DIR}/dev_*/best_model.pt"
log "  Training log:         ${LOG_FILE}"
log "  WandB project:        ${WANDB_PROJECT}"
log ""
log "Pipeline finished at $(date)"

exit 0