#!/bin/bash
# =============================================================================
# scGPT Two-Stage Training Pipeline
# Stage 1: Continual Pretraining on Norman Perturb-seq Data (15 epochs)
# Stage 2: PBMC 10K Integration Fine-tuning (15 epochs)
# =============================================================================

set -e

# =============================================================================
# 0. Environment & Path Setup
# =============================================================================
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/84c5c907-7532-424d-877d-2b5578a5a296/scGPT/code/c4d74bef-7e91-4a20-b49b-ed307fe1018f/scGPT"

# Disable wandb API key requirement - use offline mode
export WANDB_MODE="offline"
export WANDB_SILENT="true"

# cd to tutorials directory so relative paths (../save/) resolve correctly
cd "${PROJECT_ROOT}/tutorials"

echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time: $(date)"
echo "=========================================="

# =============================================================================
# 1. Auto-select GPU with lowest memory usage
# =============================================================================
echo ""
echo "=========================================="
echo "Step 1: Selecting optimal GPU"
echo "=========================================="

# Use nvidia-smi to find the GPU with maximum free memory
FREE_MEM=()
for i in $(seq 0 $(($(nvidia-smi --list-gpus | wc -l) - 1))); do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null || echo "99999")
    MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i 2>/dev/null || echo "0")
    FREE_MEM[$i]=$MEM_FREE
    echo "  GPU $i: ${MEM_USED}MiB used, ${MEM_FREE}MiB free"
done

# Find GPU with max free memory
BEST_GPU=0
MAX_FREE=0
for i in "${!FREE_MEM[@]}"; do
    if [ "${FREE_MEM[$i]}" -gt "$MAX_FREE" ]; then
        MAX_FREE="${FREE_MEM[$i]}"
        BEST_GPU=$i
    fi
done

export CUDA_VISIBLE_DEVICES=$BEST_GPU
echo "Selected GPU: ${BEST_GPU} (${MAX_FREE}MiB free)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# =============================================================================
# 2. Verify Pretrained Model Checkpoint
# =============================================================================
echo ""
echo "=========================================="
echo "Step 2: Verifying pretrained model checkpoint"
echo "=========================================="

PRETRAINED_DIR="${PROJECT_ROOT}/save/scGPT_human"
if [ ! -d "${PRETRAINED_DIR}" ] || [ ! -f "${PRETRAINED_DIR}/best_model.pt" ]; then
    echo "WARNING: Pretrained model not found at ${PRETRAINED_DIR}"
    echo "Initializing a minimal checkpoint for architecture loading..."
    mkdir -p "${PRETRAINED_DIR}"
    python3 "${PROJECT_ROOT}/scripts/init_checkpoint.py" 2>&1 | tee -a "${PROJECT_ROOT}/run_log/init_checkpoint.log"
    echo ""
    echo "Checkpoint initialization complete."
    echo "NOTE: Using randomly initialized weights (not actual pretrained)."
    echo "Training will start from scratch but with proper model architecture."
else
    echo "Pretrained model found at: ${PRETRAINED_DIR}"
    ls -la "${PRETRAINED_DIR}/"
fi

# =============================================================================
# 3. Install/Check Dependencies
# =============================================================================
echo ""
echo "=========================================="
echo "Step 3: Checking dependencies"
echo "=========================================="

# Check if pertpy is available for Norman data loading
python3 -c "import pertpy" 2>/dev/null && echo "pertpy available" || {
    echo "pertpy not found. Installing..."
    pip install pertpy 2>&1 | tail -1
}

# Check scib for evaluation metrics
python3 -c "import scib" 2>/dev/null && echo "scib available" || {
    echo "scib not found. Installing..."
    pip install scib 2>&1 | tail -1
}

# =============================================================================
# 4. Create Data Directory for Norman
# =============================================================================
echo ""
echo "=========================================="
echo "Step 4: Preparing data directories"
echo "=========================================="

mkdir -p "${PROJECT_ROOT}/data/norman"
mkdir -p "${PROJECT_ROOT}/save"

echo "Data directories ready."

# =============================================================================
# 5. Run Two-Stage Training
# =============================================================================
echo ""
echo "=========================================="
echo "Step 5: Running Two-Stage Training"
echo "=========================================="
echo ""
echo "Stage 1: Norman Continual Pretraining (15 epochs)"
echo "  - Dataset: Norman Perturb-seq (K562, various gene perturbations)"
echo "  - Loss: MLM + MVC + ECS + DAR (perturbation as pseudo-batch)"
echo "  - Learning Rate: 1e-4"
echo "  - Batch Size: 64"
echo ""
echo "Stage 2: PBMC 10K Integration Fine-tuning (15 epochs)"
echo "  - Dataset: PBMC 10K (10 batches)"
echo "  - Loss: MLM + MVC + ECS + DAR (batch labels)"
echo "  - Learning Rate: 1e-4"
echo "  - Batch Size: 64"
echo ""

# Start timing
TRAIN_START=$(date +%s)

# Run the training script with absolute path
python3 "${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" 2>&1 | tee "${PROJECT_ROOT}/run_log/training_$(date +%Y%m%d_%H%M%S).log"

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
echo ""
echo "=========================================="
echo "Training completed in ${TRAIN_DURATION} seconds"
echo "End Time: $(date)"
echo "=========================================="

# =============================================================================
# 6. Collect Results Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Step 6: Collecting results"
echo "=========================================="

# Find the most recent save directory for PBMC 10K
LATEST_SAVE_DIR=$(ls -td "${PROJECT_ROOT}/save/dev_PBMC_10K-"* 2>/dev/null | head -1)
if [ -n "${LATEST_SAVE_DIR}" ]; then
    echo "Latest save directory: ${LATEST_SAVE_DIR}"
    if [ -f "${LATEST_SAVE_DIR}/best_model.pt" ]; then
        echo "Final model saved: ${LATEST_SAVE_DIR}/best_model.pt"
    fi
    # List all saved model checkpoints
    echo "Checkpoints:"
    ls -la "${LATEST_SAVE_DIR}"/*.pt 2>/dev/null || echo "  No .pt files found"
else
    echo "Warning: No PBMC 10K save directory found."
fi

# Check Norman checkpoint
NORMAN_CKPT_DIR="${PROJECT_ROOT}/save/scGPT_norman_continual_pretrain"
if [ -d "${NORMAN_CKPT_DIR}" ]; then
    echo ""
    echo "Norman Stage 1 checkpoint:"
    ls -la "${NORMAN_CKPT_DIR}/"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Expected improvements:"
echo "  - avg_bio:  0.68 -> 0.70~0.72"
echo "  - ARI:      0.71 -> 0.73~0.75"
echo "  - NMI:      improved"
echo "  - ASW_label: improved"
echo ""
echo "Output locations:"
echo "  - Norman Stage 1 checkpoint:         ${PROJECT_ROOT}/save/scGPT_norman_continual_pretrain/"
echo "  - PBMC 10K Stage 2 final model:      ${LATEST_SAVE_DIR:-<latest_save_dir>}/"
echo "  - Training logs:                      ${PROJECT_ROOT}/run_log/"
echo ""