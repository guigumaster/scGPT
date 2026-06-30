#!/bin/bash
# =============================================================================
# scGPT Enhanced Fine-tuning Pipeline
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

# Project root directory (ABSOLUTE PATH - DO NOT CHANGE)
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/b88ba9d2-da3e-446d-a909-e27c3f575abd/scGPT/code/e129a197-dfc7-47f5-8816-32638fa1f8bd/scGPT"

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

# WandB settings
WANDB_PROJECT="scGPT"
WANDB_MODE="online"  # Change to "offline" if no internet access

# GPU selection
GPU_MEM_THRESHOLD=1000  # Minimum free memory in MiB to consider a GPU usable

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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log "ERROR: $1 is not installed. Please install it first."
        exit 1
    fi
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
check_command python3
log "Python: $(python3 --version)"

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>&1 | tee -a "${LOG_FILE}"

# Check GPU memory and select best GPU
log "Checking GPU memory availability..."
python3 -c "
import subprocess, sys
from io import StringIO
import pandas as pd

try:
    gpu_stats = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=index,memory.used,memory.free,name']).decode('utf-8')
    gpu_df = pd.read_csv(StringIO(gpu_stats), names=['index', 'memory.used', 'memory.free', 'name'], skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
    print('GPU availability:')
    print(gpu_df.to_string(index=False))
    
    # Filter GPUs with enough free memory
    threshold = ${GPU_MEM_THRESHOLD}
    usable = gpu_df[gpu_df['memory.free'] > threshold]
    if len(usable) > 0:
        best_gpu = usable.loc[usable['memory.free'].idxmax()]
        print(f'\\nSelected GPU {int(best_gpu[\"index\"])}: {best_gpu[\"name\"]} with {best_gpu[\"memory.free\"]} MiB free')
        print(f'Recommend: export CUDA_VISIBLE_DEVICES={int(best_gpu[\"index\"])}')
    else:
        print(f'\\nWARNING: No GPU has >{threshold} MiB free memory. Using default GPU 0.')
except Exception as e:
    print(f'Error checking GPU: {e}')
" 2>&1 | tee -a "${LOG_FILE}"

# Ask user to set CUDA_VISIBLE_DEVICES based on the output above
# Uncomment and set the desired GPU index after checking availability:
# export CUDA_VISIBLE_DEVICES=0

# Check dependencies
log "Checking Python dependencies..."
python3 -c "
import sys
missing = []
try:
    import torch; print(f'  torch {torch.__version__}')
except: missing.append('torch')
try:
    import scanpy; print(f'  scanpy {scanpy.__version__}')
except: missing.append('scanpy')
try:
    import scvi; print(f'  scvi-tools')
except: missing.append('scvi')
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
    print(f'\\nMissing packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
" 2>&1 | tee -a "${LOG_FILE}"

# Check model checkpoint
if [ ! -f "${MODEL_DIR}/best_model.pt" ]; then
    log "WARNING: Model checkpoint not found at ${MODEL_DIR}"
    log "Please download the pretrained scGPT model and update MODEL_DIR in this script."
    log "See https://github.com/bowang-lab/scGPT for download links."
fi

# Check Norman dataset
if [ ! -f "${NORMAN_DATA}" ]; then
    log "WARNING: Norman dataset not found at ${NORMAN_DATA}"
    log "The pipeline will skip Norman continual pre-training and proceed with direct fine-tuning."
    log "To download: https://dataverse.harvard.edu/api/access/datafile/6154020"
fi

# =============================================================================
# Step 1: Data Preparation (if needed)
# =============================================================================

print_section "Step 1: Data Preparation"

# Create tutorials data directory
mkdir -p "${PROJECT_ROOT}/tutorials/data"

# The PBMC 10K dataset is loaded automatically via scvi.data.pbmc_dataset()
# The Norman dataset should be placed at tutorials/data/norman.h5ad
if [ -f "${NORMAN_DATA}" ]; then
    log "Norman dataset found: ${NORMAN_DATA} ($(du -h ${NORMAN_DATA} | cut -f1))"
else
    log "Norman dataset not found. Will skip Norman continual pre-training."
fi

# =============================================================================
# Step 2: Training - Enhanced Fine-tuning
# =============================================================================

print_section "Step 2: Enhanced Fine-tuning with Norman Pre-training + CCE"

log "Starting enhanced training pipeline..."
log "Key improvements enabled:"
log "  - Norman continual pre-training (${NORMAN_DATA})"
log "  - Contrastive Cell Embedding (CCE) objective"
log "  - Adaptive mask ratio curriculum (0.65 -> 0.35)"
log "  - Cosine annealing LR scheduler with warm restarts"
log "  - Extended training (30 epochs)"
log "  - Early stopping on avg_bio"

# Run the enhanced training script
cd "${PROJECT_ROOT}"

# Set wandb mode
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"

log "Running: python3 tutorials/Tutorial_Integration.py"
python3 -u tutorials/Tutorial_Integration.py 2>&1 | tee -a "${LOG_FILE}"

TRAIN_EXIT_CODE=$?
if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    log "ERROR: Training failed with exit code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi
log "Training completed successfully!"

# =============================================================================
# Step 3: Evaluation
# =============================================================================

print_section "Step 3: Final Evaluation"

log "The evaluation metrics are computed automatically during training."
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
    if ls "${LATEST_SAVE}/embeddings_"*".png" 1> /dev/null 2>&1; then
        log "Evaluation UMAP plots:"
        for img in "${LATEST_SAVE}/embeddings_"*".png"; do
            log "  - $(basename ${img})"
        done
    fi
    
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
log "=== Key Metrics to Track in WandB ==="
log "  test/avg_bio     - Average of NMI, ARI, and ASW_label (higher is better)"
log "  test/ARI         - Adjusted Rand Index for clustering (higher is better)"
log "  test/NMI         - Normalized Mutual Information (higher is better)"
log "  test/ASW_label   - Cell-type ASW (higher is better)"
log "  test/PCR_batch   - PCR batch effect removal (higher is better)"
log "  test/graph_conn  - Graph connectivity (higher is better)"
log ""
log "=== Output Files ==="
log "  Best model:           ${LATEST_SAVE}/best_model.pt"
log "  Training log:         ${LOG_FILE}"
log "  WandB project:        ${WANDB_PROJECT}"
log ""
log "=== Commands for Manual Inspection ==="
log ""
log "  # View latest training log:"
log "  tail -f ${LOG_FILE}"
log ""
log "  # Test with a specific checkpoint:"
log "  python3 -c \"
import torch
from scgpt.model import TransformerModel
model = TransformerModel(...)
model.load_state_dict(torch.load('${LATEST_SAVE}/best_model.pt'))
model.eval()
print('Model loaded successfully')
\""

log ""
log "Pipeline finished at $(date)"

exit 0