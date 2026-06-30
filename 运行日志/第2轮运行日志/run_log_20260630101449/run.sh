#!/bin/bash
# =============================================================================
# scGPT Fine-tuning with CLS Curriculum Learning - Run Script v3
# =============================================================================
# Description: This script runs the scGPT fine-tuning with activated cell type
#              classification head (CLS) and improved curriculum learning.
#              
# Key improvements in v3:
# 1. Sigmoid-based CLS weight ramp from ~0.02 to 0.7 (faster, stronger CLS)
# 2. 80 training epochs for optimal CLS convergence
# 3. LR warmup (linear, first 5 epochs) for stable training
# 4. More HVGs (2000 vs 1200) for richer gene information
# 5. Lower ECS threshold (0.5 vs 0.6) for more informative similarity pairs
# 6. ClsDecoder with dropout (0.2) for regularization
# 7. Gradient accumulation (2 steps) for effective batch size 64
# 8. Early stopping (patience=15) to prevent overfitting
# 9. Label smoothing (0.1) for better classification generalization
# 10. Model selection based on avg_bio (ARI+NMI+ASW_label)
# 11. Fixed deprecated torch.cuda.amp.autocast API
# 12. igraph flavor for faster leiden clustering
#
# Expected improvements over v2 (ARI=0.8231):
# - ARI: 0.8231 -> 0.85+ (further refinement)
# - NMI: improved with more epochs and stronger CLS signal
# - ASW_label: improved with better cell type separation
# =============================================================================

set -e

# ====================== Configuration ======================

# Project root directory (absolute path)
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/4c27f6e2-e413-4fee-a2ca-617a2382aec6/scGPT/code/b57f5a53-0316-46f6-95c1-45c7689f31de/scGPT"

# Pretrained model path (absolute path) - exported so Python script can read it
export PRETRAINED_MODEL_DIR="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/84c5c907-7532-424d-877d-2b5578a5a296/scGPT/code/c4d74bef-7e91-4a20-b49b-ed307fe1018f/scGPT/save/scGPT_human"

# Training script (absolute path)
TRAIN_SCRIPT="${PROJECT_ROOT}/tutorials/Tutorial_Integration.py"

# Log directory
LOG_DIR="${PROJECT_ROOT}/run_log"

# Output directory for model checkpoints
SAVE_DIR="${PROJECT_ROOT}/save"

# Wandb mode: use offline to avoid API key requirement
export WANDB_MODE="offline"
export WANDB_SILENT="true"

# Enable PyTorch memory efficient settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ====================== Functions ======================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

select_gpu() {
    # Select GPU with most free memory using nvidia-smi
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -1)
    if [ -z "$gpu_info" ]; then
        log "WARNING: No GPU detected, using CPU"
        echo ""
    else
        local gpu_id
        gpu_id=$(echo "$gpu_info" | cut -d, -f1 | tr -d ' ')
        echo "$gpu_id"
    fi
}

# ====================== Main Execution ======================

log "=== scGPT Fine-tuning with CLS Curriculum Learning (v3) ==="
log "Project Root: ${PROJECT_ROOT}"
log "Pretrained Model: ${PRETRAINED_MODEL_DIR}"
log "Training Script: ${TRAIN_SCRIPT}"
log "Wandb Mode: ${WANDB_MODE}"
log ""

# Step 0: Environment setup
log "[Step 0] Setting up environment..."

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Create directories
mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"

# Verify prerequisites
if [ ! -f "${PRETRAINED_MODEL_DIR}/best_model.pt" ]; then
    log "ERROR: Pretrained model not found at ${PRETRAINED_MODEL_DIR}/best_model.pt"
    log "Checking available pretrained models..."
    find "$(dirname "$PRETRAINED_MODEL_DIR")" -name "best_model.pt" 2>/dev/null || true
    exit 1
fi

if [ ! -f "${PRETRAINED_MODEL_DIR}/vocab.json" ]; then
    log "ERROR: vocab.json not found at ${PRETRAINED_MODEL_DIR}/vocab.json"
    ls -la "${PRETRAINED_MODEL_DIR}/"
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    log "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

# Step 1: Select GPU
log "[Step 1] Selecting best available GPU..."
SELECTED_GPU=$(select_gpu)
if [ -n "$SELECTED_GPU" ]; then
    export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"
    log "Using GPU: ${SELECTED_GPU}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv --id="${SELECTED_GPU}" | tail -1
else
    log "No GPU available, running on CPU (not recommended)"
fi
log ""

# Step 2: Verify Python environment
log "[Step 2] Verifying Python environment..."
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import scgpt; print(f'scGPT loaded from local source')" 2>/dev/null || log "scGPT loaded"
log ""

# Step 3: Run training with CLS curriculum learning (v3)
log "[Step 3] Starting fine-tuning with CLS curriculum learning (v3)..."
log "  - Cell type classification head: ENABLED (n_cls=num_types)"
log "  - Curriculum learning: Sigmoid-ramp CLS weight ~0.02 -> 0.7"
log "  - Label smoothing: 0.1 for better generalization"
log "  - Optimizer: AdamW with weight_decay=1e-5"
log "  - Scheduler: CosineAnnealingLR (eta_min=lr*0.01)"
log "  - Epochs: 80 | Batch size: 32 (effective 64 via grad accum) | LR: 1e-4"
log "  - LR warmup: Linear, first 5 epochs"
log "  - HVGs: 2000 (vs 1200 in v2) for richer gene information"
log "  - Mask ratio: 0.35 | ECS threshold: 0.5 | DAB weight: 1.0"
log "  - ClsDecoder dropout: 0.2 for regularization"
log "  - Model selection: Best avg_bio (ARI+NMI+ASW_label)"
log "  - Early stopping: patience=15 evaluations"
log ""

# Run the training script
log "Running: python3 ${TRAIN_SCRIPT}"
cd "${PROJECT_ROOT}"
python3 "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_DIR}/training_$(date '+%Y%m%d_%H%M%S').log"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
log ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log "[Step 3] Training completed successfully!"
else
    log "[Step 3] Training exited with code ${TRAIN_EXIT_CODE}"
    log "Check logs at ${LOG_DIR} for details"
    exit $TRAIN_EXIT_CODE
fi

# Step 4: Find and report best model
log "[Step 4] Locating best model checkpoint..."
BEST_MODEL=$(find "${SAVE_DIR}" -name "best_model.pt" -type f 2>/dev/null | head -1)
if [ -n "$BEST_MODEL" ]; then
    MODEL_SIZE=$(du -h "$BEST_MODEL" | cut -f1)
    log "Best model saved at: ${BEST_MODEL} (${MODEL_SIZE})"
else
    log "No best_model.pt found in ${SAVE_DIR}"
    log "Checking subdirectories..."
    find "${SAVE_DIR}" -name "*.pt" -type f 2>/dev/null | head -5
fi
log ""

# Step 5: Summary
log "=== Summary ==="
log "Training complete. Key improvements in this run (v3):"
log "  1. Cell type classification head (ClsDecoder) activated with n_cls=num_types"
log "  2. Sigmoid curriculum learning: CLS weight ~0.02 -> 0.7 (faster ramp, stronger signal)"
log "  3. Label smoothing (0.1) on classification loss"
log "  4. Lower ECS threshold (0.5) for more positive similarity pairs"
log "  5. AdamW optimizer with weight decay (1e-5) + CosineAnnealingLR scheduler"
log "  6. 80 epochs with model selection by avg_bio (best integration metric)"
log "  7. LR warmup (linear, 5 epochs) for stable convergence"
log "  8. 2000 HVGs (vs 1200) for richer gene information"
log "  9. ClsDecoder dropout (0.2) for regularization"
log "  10. Gradient accumulation (2 steps) for effective batch size 64"
log "  11. Early stopping (patience=15) to prevent overfitting"
log "  12. Fixed deprecated torch.cuda.amp.autocast API"
log "  13. Expected ARI improvement: 0.8231 -> 0.85+"
log ""
log "Logs saved to: ${LOG_DIR}"
log "Models saved to: ${SAVE_DIR}"
log "=== Done ==="