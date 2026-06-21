#!/bin/bash
# ============================================================
# scGPT CCE + AdaDAR Fusion Strategy - Training & Evaluation Script
# 
# This script implements:
#   1. Environment Setup
#   2. Data Preparation
#   3. Training with CCE + AdaDAR + Dynamic ECS
#   4. Evaluation and Validation
#   5. Result Logging
# ============================================================

set -euo pipefail

# ---------- Project Root ----------
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/e8ff47cf-dde4-4964-a593-2d5a08ce742d/scGPT/code/8f670df0-abab-47a5-880c-78b22efe5f50/scGPT"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${RUN_LOG_DIR}"

# ---------- 1. Environment Verification ----------
echo "[$(date)] ===== Environment Setup =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Check Python version
python --version 2>&1 | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Check CUDA availability
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
" 2>&1 | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# ---------- 2. Create Pretrained Model Directory ----------
PRETRAINED_DIR="${PROJECT_ROOT}/examples/save/scGPT_bc"
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "[$(date)] Creating pretrained model directory: $PRETRAINED_DIR" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    mkdir -p "$PRETRAINED_DIR"
fi

# ---------- 3. Download Pretrained Weights (if not exists) ----------
PRETRAINED_FILE="${PRETRAINED_DIR}/best_model.pt"
if [ ! -f "$PRETRAINED_FILE" ]; then
    echo "[$(date)] Downloading pretrained scGPT whole-human checkpoint..." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    # Option A: Download from Google Drive (requires gdown)
    # pip install gdown
    # gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O "$PRETRAINED_DIR"
    
    # Option B: Use a local or alternative source - adjust as needed
    echo "[$(date)] WARNING: Pretrained weights not found at $PRETRAINED_FILE" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "[$(date)] Please download the checkpoint from the scGPT model zoo manually." | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "[$(date)] Expected files:" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "  - ${PRETRAINED_DIR}/best_model.pt" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "  - ${PRETRAINED_DIR}/args.json" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
    echo "  - ${PRETRAINED_DIR}/vocab.json" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
fi

# ---------- 4. Install Dependencies ----------
echo "[$(date)] ===== Installing Dependencies =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Install core dependencies
pip install --quiet scgpt "flash-attn<1.0.5" "orbax<0.1.8" 2>&1 | tee -a "${RUN_LOG_DIR}/pip_install_${TIMESTAMP}.log"

# Install evaluation dependencies
pip install --quiet scib wandb 2>&1 | tee -a "${RUN_LOG_DIR}/pip_install_${TIMESTAMP}.log"

# ---------- 5. Training with CCE + AdaDAR ----------
echo "[$(date)] ===== Starting Training (CCE + AdaDAR + Dynamic ECS) =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# Default hyperparameters for CCE + AdaDAR fusion strategy
EPOCHS=30
BATCH_SIZE=64
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
WANDB_MODE="${WANDB_MODE:-online}"  # set to "offline" if no internet

# Run the finetune integration script
python -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
    --seed 42 \
    --dataset_name "PBMC_10K" \
    --do_train True \
    --load_model "${PRETRAINED_DIR}" \
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

# ---------- 7. Run Ablation Studies (Optional) ----------
# To compare with baseline, one can run without CCE and AdaDAR:
# echo "[$(date)] ===== Running Baseline (no CCE, no AdaDAR) =====" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
# python -u "${PROJECT_ROOT}/examples/finetune_integration.py" \
#     --CCE False --cce_weight 0.0 \
#     --ada_dar False \
#     --ecs_dynamic False \
#     ... (other params same as above) \
#     2>&1 | tee -a "${RUN_LOG_DIR}/train_baseline_${TIMESTAMP}.log"

# ---------- 8. Generate Summary ----------
echo "[$(date)] ====== Training Summary ======" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
echo "Project Root: ${PROJECT_ROOT}" | tee -a "${RUN_LOG_DIR}/run_${TIMESTAMP}.log"
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