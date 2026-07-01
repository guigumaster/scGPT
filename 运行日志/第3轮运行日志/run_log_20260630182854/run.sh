#!/bin/bash
# =============================================================================
# scGPT Two-Stage Training Pipeline (Optimized v4)
# Stage 1: Continual Pretraining on Norman Perturb-seq Data (8 epochs, 70% subsample)
# Stage 2: PBMC 10K Integration Fine-tuning (20 epochs)
#
# Key optimization changes from v3 (based on log analysis of 2026-06-30 run):
#   - Reduced dab_weight 1.0->0.5, ecs_thres 0.8->0.5 (less aggressive batch correction)
#   - Increased n_hvg 1200->1500, norman_n_hvg 1200->1500 (more biological signal)
#   - Increased norman_epochs 6->8, subsample_ratio 0.5->0.7 (more pretraining data)
#   - Added gradient accumulation (2 steps) for more stable training
#   - Reduced mask_ratio 0.4->0.35 (easier learning with more observed tokens)
#   - Reduced lr 1e-4->5e-5 for PBMC stage (stable fine-tuning)
#   - Added spawn multiprocessing method to fix fork warnings
#   - Increased epochs 15->20 for better convergence
# =============================================================================

set -e

# =============================================================================
# 0. Environment & Path Setup
# =============================================================================
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/84c5c907-7532-424d-877d-2b5578a5a296/scGPT/code/c4d74bef-7e91-4a20-b49b-ed307fe1018f/scGPT"

# Disable wandb API key requirement - use offline mode
export WANDB_MODE="offline"
export WANDB_SILENT="true"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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
# 4. Create Data Directories
# =============================================================================
echo ""
echo "=========================================="
echo "Step 4: Preparing data directories"
echo "=========================================="

mkdir -p "${PROJECT_ROOT}/data/norman"
mkdir -p "${PROJECT_ROOT}/save"
mkdir -p "${PROJECT_ROOT}/run_log"

echo "Data directories ready."

# =============================================================================
# 5. Run Two-Stage Training (Optimized v4)
# =============================================================================
echo ""
echo "=========================================="
echo "Step 5: Running Two-Stage Training (Optimized v4)"
echo "=========================================="
echo ""
echo "Stage 1: Norman Continual Pretraining (up to 8 epochs, 70% subsample)"
echo "  - Dataset: Norman Perturb-seq (K562, subsampled to ~78K cells)"
echo "  - Loss: MLM + MVC + ECS + DAR (perturbation as pseudo-batch)"
echo "  - Learning Rate: 2e-4 (faster convergence)"
echo "  - Batch Size: 128, Gradient Accumulation: 2 steps (effective batch 256)"
echo "  - HVGs: 1500 (more biological signal)"
echo "  - DataLoader Workers: 4, spawn multiprocessing"
echo "  - Early Stopping: patience=3"
echo "  - Expected time: ~20-25 minutes"
echo ""
echo "Stage 2: PBMC 10K Integration Fine-tuning (20 epochs)"
echo "  - Dataset: PBMC 10K (10 batches)"
echo "  - Loss: MLM + MVC + ECS + DAR (batch labels)"
echo "  - Learning Rate: 5e-5 (stable fine-tuning)"
echo "  - Batch Size: 128, Gradient Accumulation: 2 steps"
echo "  - dab_weight: 0.5, ecs_thres: 0.5 (balanced batch correction)"
echo "  - mask_ratio: 0.35 (more observed tokens)"
echo "  - HVGs: 1500 (more biological signal)"
echo "  - DataLoader Workers: 4"
echo "  - Expected time: ~30-35 minutes"
echo ""
echo "Expected total time: ~50-60 minutes (well within 3-hour limit)"
echo ""

# Record GPU memory before training
echo "GPU memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -i $BEST_GPU

# Start timing
TRAIN_START=$(date +%s)

# Run the training script with absolute path
# Use timeout of 2.5 hours (9000 seconds) to ensure we don't exceed the 3-hour limit
TIMEOUT_SEC=9000
echo "Setting timeout: ${TIMEOUT_SEC}s (2.5 hours)"
echo ""

timeout ${TIMEOUT_SEC} python3 -u "${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" 2>&1 | tee "${PROJECT_ROOT}/run_log/training_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
echo "=========================================="
echo "Training completed in ${TRAIN_DURATION} seconds"
echo "Exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 124 ]; then
    echo "WARNING: Training timed out after ${TIMEOUT_SEC}s (2.5 hours)"
    echo "The timeout limit was set to leave a 30-minute buffer within the 3-hour limit."
fi

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
    
    # Check if evaluation results are available
    if ls "${LATEST_SAVE_DIR}"/embeddings_*.png 1>/dev/null 2>&1; then
        echo ""
        echo "UMAP plots generated:"
        ls -la "${LATEST_SAVE_DIR}"/embeddings_*.png
    fi
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

# Record GPU memory after training
echo ""
echo "GPU memory after training:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -i $BEST_GPU

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Expected improvements (with Norman continual pretraining):"
echo "  - avg_bio:  0.68 -> 0.70~0.75"
echo "  - ARI:      0.02 -> 0.35~0.55 (targeting significant improvement)"
echo "  - NMI:      improved"
echo "  - ASW_label: improved"
echo ""
echo "Output locations:"
echo "  - Norman Stage 1 checkpoint:         ${PROJECT_ROOT}/save/scGPT_norman_continual_pretrain/"
echo "  - PBMC 10K Stage 2 final model:      ${LATEST_SAVE_DIR:-<latest_save_dir>}/"
echo "  - Training logs:                      ${PROJECT_ROOT}/run_log/"
echo "  - Pipeline script:                   ${PROJECT_ROOT}/run_log/run.sh"
echo ""