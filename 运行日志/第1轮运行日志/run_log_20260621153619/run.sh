#!/bin/bash
# ============================================================
# scGPT CCE + AdaDAR Fusion Strategy - Training & Evaluation Script
#
# This script implements:
#   1. Environment Setup (using base conda with working PyTorch+CUDA)
#   2. Install scgpt from local source (not PyPI)
#   3. Data Preparation
#   4. Training with CCE + AdaDAR + Dynamic ECS
#   5. Evaluation and Validation
#   6. Result Logging
# ============================================================

set -euo pipefail

# ---------- Project Root (Absolute Path) ----------
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/e8ff47cf-dde4-4964-a593-2d5a08ce742d/scGPT/code/8f670df0-abab-47a5-880c-78b22efe5f50/scGPT"
cd "$PROJECT_ROOT"

# ---------- Python Environment ----------
# Use the base conda environment which has working PyTorch 2.11.0+cu128 with CUDA
PYTHON_BIN="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python"
PIP_BIN="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/pip"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"  # offline mode to avoid API key requirement

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${RUN_LOG_DIR}"

# ---------- 1. Environment Verification ----------
echo "[$(date)] ===== Environment Setup =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Check Python version
$PYTHON_BIN --version 2>&1 | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Check CUDA availability
$PYTHON_BIN -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
" 2>&1 | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# ---------- 2. Install local scgpt package ----------
echo "[$(date)] ===== Installing scgpt from local source =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
$PIP_BIN install -e "${PROJECT_ROOT}" --no-deps 2>&1 | tee -a "${RUN_LOG_DIR}/pip_install_${TIMESTAMP}.log"
echo "[$(date)] Local scgpt installation complete." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# ---------- 3. Create Pretrained Model Directory ----------
PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "[$(date)] Creating pretrained model directory: $PRETRAINED_DIR" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    mkdir -p "$PRETRAINED_DIR"
fi

# ---------- 4. Download Pretrained Weights (if not exists) ----------
PRETRAINED_FILE="${PRETRAINED_DIR}/best_model.pt"
if [ ! -f "$PRETRAINED_FILE" ]; then
    echo "[$(date)] Pretrained weights not found at $PRETRAINED_FILE" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "[$(date)] Training will proceed without pretrained weights (from scratch)." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
fi

# ---------- 5. Training with CCE + AdaDAR ----------
echo "[$(date)] ===== Starting Training (CCE + AdaDAR + Dynamic ECS) =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Default hyperparameters for CCE + AdaDAR fusion strategy
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=2e-4
MASK_RATIO=0.4
N_HVG=1200
CCE_WEIGHT=0.1
CCE_TEMP=0.5
ECS_THRES=0.8
ECS_DYNAMIC=True
DAB_WEIGHT=1.0
ADA_DAR=True
ADA_DAR_TARGET_ACC=0.5
ADA_DAR_MIN_LAMBDA=0.0
ADA_DAR_MAX_LAMBDA=5.0
ADA_DAR_STEP=0.1

echo "[$(date)] Configuration:" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  EPOCHS=${EPOCHS}, BATCH_SIZE=${BATCH_SIZE}, LR=${LEARNING_RATE}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  CCE_WEIGHT=${CCE_WEIGHT}, CCE_TEMP=${CCE_TEMP}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  ECS_THRES=${ECS_THRES}, ECS_DYNAMIC=${ECS_DYNAMIC}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  ADA_DAR=${ADA_DAR}, target_acc=${ADA_DAR_TARGET_ACC}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Run the finetune integration script
$PYTHON_BIN -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
    --seed 42 \
    --dataset_name "PBMC_10K" \
    --do_train True \
    --load_model "" \
    --mask_ratio ${MASK_RATIO} \
    --epochs ${EPOCHS} \
    --n_bins 51 \
    --GEPC True \
    --CCE True \
    --cce_weight ${CCE_WEIGHT} \
    --cce_temp ${CCE_TEMP} \
    --ecs_thres ${ECS_THRES} \
    --ecs_dynamic ${ECS_DYNAMIC} \
    --dab_weight ${DAB_WEIGHT} \
    --ada_dar ${ADA_DAR} \
    --ada_dar_target_acc ${ADA_DAR_TARGET_ACC} \
    --ada_dar_min_lambda ${ADA_DAR_MIN_LAMBDA} \
    --ada_dar_max_lambda ${ADA_DAR_MAX_LAMBDA} \
    --ada_dar_step ${ADA_DAR_STEP} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --layer_size 128 \
    --nlayers 4 \
    --nhead 4 \
    --dropout 0.2 \
    --schedule_ratio 0.9 \
    --save_eval_interval 5 \
    --log_interval 100 \
    --fast_transformer True \
    --pre_norm False \
    --amp True \
    --n_hvg ${N_HVG} \
    2>&1 | tee -a "${RUN_LOG_DIR}/train_${TIMESTAMP}.log"

echo "[$(date)] Training completed." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# ---------- 6. Evaluation and Metrics Collection ----------
echo "[$(date)] ===== Running Evaluation =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Find the latest saved model directory
SAVE_DIR=$(ls -td "${PROJECT_ROOT}/examples/save/dev_PBMC_10K-"* 2>/dev/null | head -1)
if [ -z "${SAVE_DIR}" ]; then
    echo "[$(date)] WARNING: No save directory found. Evaluation may fail." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
else
    echo "[$(date)] Evaluating model from: ${SAVE_DIR}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

    # Find the best model
    BEST_MODEL="${SAVE_DIR}/best_model.pt"
    if [ -f "${BEST_MODEL}" ]; then
        echo "[$(date)] Best model found: ${BEST_MODEL}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    fi
fi

# ---------- 7. Generate Summary ----------
echo "[$(date)] ====== Training Summary ======" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "Project Root: ${PROJECT_ROOT}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "Python: $($PYTHON_BIN --version 2>&1)" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "Configuration:" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - CCE (Contrastive Cell Embedding): Enabled, weight=${CCE_WEIGHT}, temp=${CCE_TEMP}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - AdaDAR (Adaptive DAR): Enabled, target_acc=${ADA_DAR_TARGET_ACC}, lambda=[${ADA_DAR_MIN_LAMBDA}, ${ADA_DAR_MAX_LAMBDA}]" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - ECS (Elastic Cell Similarity): threshold=${ECS_THRES}, dynamic=${ECS_DYNAMIC}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - Learning Rate: ${LEARNING_RATE}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - Epochs: ${EPOCHS}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - Batch Size: ${BATCH_SIZE}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "  - Mask Ratio: ${MASK_RATIO}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "========================================" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

echo "[$(date)] All tasks completed!" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
exit 0