#!/bin/bash
# ============================================================================
# scGPT Norman-Enhanced Pipeline: Run Script
# ============================================================================
# 
# This script implements the complete pipeline:
#   Step 1: Environment setup
#   Step 2: Model availability check
#   Step 3: Data preparation
#   Step 4: (Optional) Continual pretraining on Norman Perturb-seq data
#   Step 5: Fine-tune on BMMC/PBMC 10K integration task with ARI-boosting enhancements
#   Step 6: Baseline fine-tuning (for comparison)
#   Step 7: Evaluation summary
#
# Usage:
#   bash run_log/run.sh                          # Full pipeline
#   bash run_log/run.sh --skip-norman             # Skip continual pretraining
#   bash run_log/run.sh --norman-only             # Only do continual pretraining
#   bash run_log/run.sh --data-path <path>        # Custom Norman data path
# ============================================================================

set -e  # Exit on error
set -o pipefail

# ============================================================================
# Configuration - All paths are absolute as required
# ============================================================================

# Project root directory (absolute path as required by the task)
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/c6ecd5c3-3a17-42c8-9027-976cd4da0d12/scGPT/code/1aed9f9f-0527-4acf-bdea-155f95135436/scGPT"

# Python environment
PYTHON="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python"

# Data directories (absolute paths)
DATA_DIR="${PROJECT_ROOT}/data"
NORMAN_DATA_DIR="${DATA_DIR}/norman"
SAVE_DIR="${PROJECT_ROOT}/save"

# Model paths (absolute paths)
ORIGINAL_MODEL="${SAVE_DIR}/scGPT_human"
NORMAN_ENHANCED_DIR="${SAVE_DIR}/norman_enhanced"

# Training configuration
NORMAN_EPOCHS=20
FINETUNE_EPOCHS=30
BATCH_SIZE=32
FINETUNE_BATCH_SIZE=64
NORMAN_LR=5e-5
FINETUNE_LR=1e-4
SEED=42

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/run_log"
PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Parse command-line arguments
SKIP_NORMAN=false
NORMAN_ONLY=false
NORMAN_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-norman) SKIP_NORMAN=true; shift ;;
        --norman-only) NORMAN_ONLY=true; shift ;;
        --data-path) NORMAN_DATA_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Step 0: Environment Setup & Logging
# ============================================================================

exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "================================================================"
echo "  scGPT Norman-Enhanced Pipeline"
echo "================================================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time: $(date)"
echo "Python: ${PYTHON}"
echo "================================================================"

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Set PYTHONPATH to include project root (fixes ModuleNotFoundError for scgpt)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "PYTHONPATH=${PYTHONPATH}"

# Log system info
echo "Python: $(${PYTHON} --version)"
echo "PyTorch: $(${PYTHON} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not available")"
echo "CUDA available: $(${PYTHON} -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")"
echo "GPU count: $(${PYTHON} -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")"
echo ""

# ============================================================================
# Step 1: Data & Model Directories Preparation
# ============================================================================

echo "---------------------------------------------------------------"
echo "  Step 1: Data & Model Directories Preparation"
echo "---------------------------------------------------------------"

mkdir -p "${NORMAN_DATA_DIR}"
mkdir -p "${SAVE_DIR}"

# Check for original scGPT model
MODEL_AVAILABLE=false
if [ -d "${ORIGINAL_MODEL}" ]; then
    if [ -f "${ORIGINAL_MODEL}/best_model.pt" ] && [ -f "${ORIGINAL_MODEL}/vocab.json" ]; then
        MODEL_AVAILABLE=true
        echo "✓ Found original scGPT model: ${ORIGINAL_MODEL}"
        ls -la "${ORIGINAL_MODEL}/"
    else
        echo "WARNING: Directory ${ORIGINAL_MODEL} exists but model files missing."
        echo "  Expected: best_model.pt and vocab.json"
    fi
else
    echo "WARNING: Original scGPT model not found at ${ORIGINAL_MODEL}"
    echo "  The pipeline will still run with the enhanced Tutorial_Integration.py"
    echo "  which can train from scratch (random init) for testing."
    echo "  For best results, download the whole-human model from:"
    echo "  https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"
    echo "  And extract to: ${ORIGINAL_MODEL}"
fi
echo ""

# ============================================================================
# Step 2: Verify scgpt import works
# ============================================================================

echo "---------------------------------------------------------------"
echo "  Step 2: Verify scgpt Import"
echo "---------------------------------------------------------------"

${PYTHON} -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
try:
    import scgpt as scg
    print(f'✓ scgpt {scg.__version__} imported successfully')
    print(f'  Package location: {scg.__file__}')
except ImportError as e:
    print(f'✗ Failed to import scgpt: {e}')
    print(f'  PYTHONPATH={sys.path[:3]}')
    sys.exit(1)
"
echo ""

# ============================================================================
# Step 3: Norman Perturb-seq Continual Pretraining
# ============================================================================

if [ "${SKIP_NORMAN}" = false ]; then
    echo "---------------------------------------------------------------"
    echo "  Step 3: Norman Perturb-seq Continual Pretraining"
    echo "---------------------------------------------------------------"
    echo "  Dataset: Norman Perturb-seq (105 CRISPR perturbations)"
    echo "  Objectives: MLM + GEPC/MVC (no ECS, no DAB)"
    echo "  Epochs: ${NORMAN_EPOCHS}"
    echo "  Learning rate: ${NORMAN_LR}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "---------------------------------------------------------------"
    
    # Build command with absolute paths
    NORMAN_SCRIPT="${PROJECT_ROOT}/tutorials/scripts/norman_continual_pretrain.py"
    NORMAN_CMD="${PYTHON} ${NORMAN_SCRIPT}"
    NORMAN_CMD="${NORMAN_CMD} --load_model ${ORIGINAL_MODEL}"
    NORMAN_CMD="${NORMAN_CMD} --save_dir ${NORMAN_ENHANCED_DIR}"
    NORMAN_CMD="${NORMAN_CMD} --epochs ${NORMAN_EPOCHS}"
    NORMAN_CMD="${NORMAN_CMD} --batch_size ${BATCH_SIZE}"
    NORMAN_CMD="${NORMAN_CMD} --lr ${NORMAN_LR}"
    NORMAN_CMD="${NORMAN_CMD} --seed ${SEED}"
    
    if [ -n "${NORMAN_DATA_PATH}" ]; then
        NORMAN_CMD="${NORMAN_CMD} --data_path ${NORMAN_DATA_PATH}"
    fi
    
    echo "Running: ${NORMAN_CMD}"
    echo ""
    
    # Run Norman continual pretraining from project root
    cd "${PROJECT_ROOT}"
    eval "${NORMAN_CMD}"
    
    echo ""
    if [ $? -eq 0 ]; then
        echo "✓ Norman continual pretraining complete!"
    else
        echo "⚠ Norman continual pretraining encountered issues (see above)."
        echo "  Continuing pipeline without Norman enhancement."
    fi
    echo ""
else
    echo "---------------------------------------------------------------"
    echo "  Step 3: Skipped (--skip-norman flag set)"
    echo "---------------------------------------------------------------"
    echo ""
fi

# Exit if only Norman pretraining was requested
if [ "${NORMAN_ONLY}" = true ]; then
    echo "================================================================"
    echo "  Norman-only mode. Pipeline complete."
    echo "================================================================"
    exit 0
fi

# ============================================================================
# Step 4: Fine-tuning on BMMC/PBMC 10K Integration Task (Enhanced)
# ============================================================================

echo "---------------------------------------------------------------"
echo "  Step 4: Enhanced Fine-tuning on Integration Task"
echo "---------------------------------------------------------------"
echo "  Dataset: PBMC_10K (BMMC multi-omics integration)"
echo "  Enhanced with: CCE contrastive loss, CosineAnnealingLR,"
echo "                 weight decay, gradient clipping"
echo "  Epochs: ${FINETUNE_EPOCHS}"
echo "  Learning rate: ${FINETUNE_LR}"
echo "  Batch size: ${FINETUNE_BATCH_SIZE}"
echo "---------------------------------------------------------------"

# Build command with absolute paths
INTEGRATION_SCRIPT="${PROJECT_ROOT}/tutorials/Tutorial_Integration.py"
FINETUNE_CMD="${PYTHON} ${INTEGRATION_SCRIPT}"
FINETUNE_CMD="${FINETUNE_CMD} --load_model ${ORIGINAL_MODEL}"
FINETUNE_CMD="${FINETUNE_CMD} --epochs ${FINETUNE_EPOCHS}"
FINETUNE_CMD="${FINETUNE_CMD} --lr ${FINETUNE_LR}"
FINETUNE_CMD="${FINETUNE_CMD} --batch_size ${FINETUNE_BATCH_SIZE}"
FINETUNE_CMD="${FINETUNE_CMD} --seed ${SEED}"
FINETUNE_CMD="${FINETUNE_CMD} --dab_weight 1.0"

# If Norman-enhanced checkpoint exists, use it
if [ -d "${NORMAN_ENHANCED_DIR}" ]; then
    # Find the latest norman_enhanced checkpoint
    LATEST_NORMAN=$(ls -td "${NORMAN_ENHANCED_DIR}"*/ 2>/dev/null | head -1)
    if [ -n "${LATEST_NORMAN}" ] && [ -f "${LATEST_NORMAN}best_model.pt" ]; then
        echo "Found Norman-enhanced checkpoint: ${LATEST_NORMAN}"
        FINETUNE_CMD="${FINETUNE_CMD} --use_norman"
        FINETUNE_CMD="${FINETUNE_CMD} --norman_model_path ${LATEST_NORMAN}"
    elif [ -f "${NORMAN_ENHANCED_DIR}/best_model.pt" ]; then
        echo "Found Norman-enhanced checkpoint: ${NORMAN_ENHANCED_DIR}"
        FINETUNE_CMD="${FINETUNE_CMD} --use_norman"
        FINETUNE_CMD="${FINETUNE_CMD} --norman_model_path ${NORMAN_ENHANCED_DIR}"
    fi
fi

echo "Running: ${FINETUNE_CMD}"
echo ""

# Run fine-tuning from project root
cd "${PROJECT_ROOT}"
eval "${FINETUNE_CMD}"

echo ""
echo "✓ Enhanced fine-tuning complete!"
echo ""

# ============================================================================
# Step 5: Baseline Fine-tuning (without Norman, for comparison)
# ============================================================================

echo "---------------------------------------------------------------"
echo "  Step 5: Baseline Fine-tuning (without Norman enhancement)"
echo "---------------------------------------------------------------"
echo "  Running baseline with original scGPT model for comparison..."
echo "---------------------------------------------------------------"

BASELINE_CMD="${PYTHON} ${INTEGRATION_SCRIPT}"
BASELINE_CMD="${BASELINE_CMD} --load_model ${ORIGINAL_MODEL}"
BASELINE_CMD="${BASELINE_CMD} --epochs ${FINETUNE_EPOCHS}"
BASELINE_CMD="${BASELINE_CMD} --lr ${FINETUNE_LR}"
BASELINE_CMD="${BASELINE_CMD} --batch_size ${FINETUNE_BATCH_SIZE}"
BASELINE_CMD="${BASELINE_CMD} --seed ${SEED}"
BASELINE_CMD="${BASELINE_CMD} --dab_weight 1.0"

echo "Running: ${BASELINE_CMD}"
echo ""

cd "${PROJECT_ROOT}"
eval "${BASELINE_CMD}"

echo ""
echo "✓ Baseline fine-tuning complete!"
echo ""

# ============================================================================
# Step 6: Summary
# ============================================================================

echo "================================================================"
echo "  Pipeline Complete!"
echo "================================================================"
echo "End Time: $(date)"
echo ""
echo "Output locations:"
echo "  Pipeline log:      ${PIPELINE_LOG}"
echo "  Norman-enhanced model: ${SAVE_DIR}/norman_enhanced_*/"
echo "  Fine-tuned model:      ${PROJECT_ROOT}/tutorials/save/dev_PBMC_10K-*/"
echo "  Logs:                  ${PROJECT_ROOT}/tutorials/save/dev_PBMC_10K-*/run.log"
echo ""
echo "Key metrics to compare (from wandb logs or run.log):"
echo "  - ARI_cluster/label (primary metric for improvement)"
echo "  - NMI_cluster/label"
echo "  - ASW_label"
echo "  - avg_bio (composite score)"
echo ""
echo "Enhancements applied for ARI improvement:"
echo "  1. Norman Perturb-seq continual pretraining (105 CRISPR perturbations)"
echo "  2. CCE contrastive cell embedding loss"
echo "  3. Cosine annealing LR scheduler (eta_min=1e-6)"
echo "  4. AdamW optimizer with weight_decay=1e-5"
echo "  5. Gradient clipping (max_norm=1.0)"
echo "  6. 30 epochs for thorough convergence"
echo "================================================================"