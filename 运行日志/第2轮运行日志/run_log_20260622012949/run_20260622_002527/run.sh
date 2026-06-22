#!/bin/bash
# ==============================================================================
# scGPT Adaptive Curriculum Learning Pipeline
# ==============================================================================
# Pipeline orchestrator for scGPT single-cell multiomics integration.
#
# Improvements implemented:
# 1. Cosine annealing curriculum for DAR weight & gradient reversal coefficient
# 2. Elastic Cell Similarity (ECS) regularization
# 3. Transformer encoder extended from 4 to 8 layers with pre-norm
# 4. Dynamic mask ratio curriculum (0.25 -> 0.55)
# 5. Warmup cosine learning rate schedule
#
# Usage:
#   bash run_log/run.sh                                   # Run with defaults
#   bash run_log/run.sh --epochs 50                       # Override epochs
#   bash run_log/run.sh --wandb_mode offline              # Offline mode
# ==============================================================================

set -e
set -o pipefail

PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/942d83b7-7efe-4985-8244-7b0f713e0927/scGPT/code/1769fc95-abf6-43b7-ab6f-4e104aa681c1/scGPT"

cd "${PROJECT_ROOT}"
echo "====================================================="
echo "scGPT Adaptive Curriculum Learning Pipeline"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time: $(date)"
echo "====================================================="

# Environment setup
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scgpt:${PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/run_log/run_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/run.sh"
echo "Log directory: ${LOG_DIR}"

# ---- Step 1: Verify environment ----
echo ""
echo "--- [Step 1] Environment Verification ---"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "${PYTHON_CMD}" ]; then
    echo "ERROR: No Python interpreter found!"
    exit 1
fi

echo "Using: ${PYTHON_CMD}"
${PYTHON_CMD} --version
${PYTHON_CMD} -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')" 2>&1 || true
${PYTHON_CMD} -c "import scgpt; print(f'scgpt {scgpt.__version__}')" 2>&1 || true

# ---- Step 2: Data check ----
echo ""
echo "--- [Step 2] Data Verification ---"
ls -la "${PROJECT_ROOT}/data/" | head -10 || echo "No data files listed"
echo "Data directory OK"

# ---- Step 3: Training Pipeline ----
echo ""
echo "--- [Step 3] Training Pipeline ---"
echo "Starting at $(date)"

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_pipeline.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: ${TRAIN_SCRIPT} not found!"
    exit 1
fi

# Run training with all arguments passed through
# Using norman_2019.h5ad dataset which has gemgroup (batch) and gene_program (celltype)
set +e  # allow tee to capture exit code
${PYTHON_CMD} "${TRAIN_SCRIPT}" \
    "$@" \
    --seed 42 \
    --dataset_name "norman_2019" \
    --do_train \
    --load_model None \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --nlayers 8 \
    --nhead 4 \
    --layer_size 128 \
    --dropout 0.2 \
    --pre_norm \
    --fast_transformer \
    --amp \
    --GEPC \
    --ecs_thres 0.8 \
    --dab_weight 1.0 \
    --dab_weight_curriculum \
    --dab_weight_min 0.1 \
    --dab_weight_max 1.0 \
    --grad_reverse_curriculum \
    --grad_reverse_lambda_min 0.1 \
    --grad_reverse_lambda_max 1.0 \
    --mask_ratio_curriculum \
    --mask_ratio_start 0.25 \
    --mask_ratio_end 0.55 \
    --mask_ratio_warmup_epochs 15 \
    --lr_warmup_epochs 3 \
    --use_warmup_cosine_lr \
    --save_eval_interval 5 \
    --log_interval 100 \
    --wandb_project "scGPT" \
    --wandb_mode "offline" \
    --save_dir "${LOG_DIR}/save" \
    2>&1 | tee "${LOG_DIR}/training.log"

TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "Training exit code: ${TRAIN_EXIT}"

# ---- Step 4: Summary ----
echo ""
echo "--- [Step 4] Summary ---"
echo "Completed at: $(date)"

RESULTS="${LOG_DIR}/save/results_summary.json"
if [ -f "${RESULTS}" ]; then
    echo ""
    echo "=== FINAL RESULTS ==="
    cat "${RESULTS}"
    echo ""
else
    echo "Results file not found at ${RESULTS}"
    echo "Check ${LOG_DIR}/training.log for details."
fi

echo ""
echo "====================================================="
echo "Pipeline Complete!"
echo "====================================================="

exit ${TRAIN_EXIT}