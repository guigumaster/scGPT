#!/bin/bash
# =============================================================================
# scGPT Fine-tuning Run Script
# Strategy: CLS Cell Type Classification + ECS Activation + Tuned DAB Weight
# =============================================================================
set -e

# ----------------------------- Configuration ---------------------------------
# Project root directory (absolute path)
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/5591f1bd-78af-49c0-9cd4-157685035527/scGPT/code/26e1ed7a-c645-4a53-99c7-855bcaf49850/scGPT"

# Pretrained model source directory
PRETRAINED_SRC="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/test_example/data/scgpt_integration/checkpoints/scGPT_human"

# GPU configuration: use GPU 0 (the only available H20 GPU, 95.1 GB)
export CUDA_VISIBLE_DEVICES=0

# Wandb: use offline mode to avoid hanging (no API key configured)
export WANDB_MODE=offline

# -------------------------- Prepare Environment ------------------------------
cd "${PROJECT_ROOT}"
echo "=== Working directory: $(pwd) ==="
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="

# Create save directory for pretrained model
SAVE_DIR="${PROJECT_ROOT}/save/scGPT_human"
mkdir -p "${SAVE_DIR}"

# Copy pretrained model files if not already present
if [ ! -f "${SAVE_DIR}/best_model.pt" ]; then
    echo "=== Copying pretrained model from ${PRETRAINED_SRC} to ${SAVE_DIR} ==="
    cp "${PRETRAINED_SRC}/best_model.pt"     "${SAVE_DIR}/best_model.pt"
    cp "${PRETRAINED_SRC}/vocab.json"        "${SAVE_DIR}/vocab.json"
    cp "${PRETRAINED_SRC}/args.json"         "${SAVE_DIR}/args.json"
    echo "=== Pretrained model copied successfully ==="
else
    echo "=== Pretrained model already exists at ${SAVE_DIR} ==="
fi

# Verify pretrained model files
echo "=== Verifying pretrained model files ==="
ls -lh "${SAVE_DIR}/"
echo ""

# ----------------------------- Run Training ----------------------------------
# Change to the tutorials directory where Tutorial_Integration.py is located
cd "${PROJECT_ROOT}/tutorials"
echo "=== Running training from: $(pwd) ==="

# Run the integration fine-tuning script
# Key hyperparameter changes for the three-in-one strategy:
#   ecs_thres=0.4   (was 0.8)  - Activate Elastic Cell Similarity with lower threshold
#   dab_weight=0.5  (was 1.0)  - Reduce DAR weight for better biological conservation
#   CLS=True                    - Enable cell type classification supervision (via n_cls=num_types)
python3 "${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" 2>&1 | tee "${PROJECT_ROOT}/run_log/training_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training completed ==="

# -------------------------- Print Summary ------------------------------------
echo ""
echo "====================================================="
echo "  Training Summary"
echo "====================================================="
echo "  Project Root : ${PROJECT_ROOT}"
echo "  Pretrained   : ${PRETRAINED_SRC}"
echo "  Strategy     : CLS + ECS(0.4) + DAB(0.5)"
echo "  Logs         : ${PROJECT_ROOT}/run_log/"
echo "  Outputs      : ${PROJECT_ROOT}/tutorials/save/"
echo "====================================================="