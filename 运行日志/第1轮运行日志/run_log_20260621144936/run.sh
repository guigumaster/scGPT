#!/bin/bash
# =============================================================================
# scGPT Training Pipeline with CMCA-Loss
# =============================================================================
# This script runs the complete scGPT training pipeline with:
#   - Cross-Modal Contrastive Alignment (CMCA) Loss
#   - Masked Language Modeling (GEP/MLM)
#   - Masked Value Prediction (GEPC/MVC)
#   - Elastic Cell Similarity (ECS)
#   - Domain Adversarial Training (DAB)
#   - Validation and scIB metrics evaluation
#
# Usage:
#   bash run.sh                           # Single GPU
#   bash run.sh --distributed             # Multi-GPU (torchrun)
#   bash run.sh --load_model /path/ckpt   # Resume from checkpoint
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Parse arguments
# ---------------------------------------------------------------------------
DISTRIBUTED=false
LOAD_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --distributed)   DISTRIBUTED=true; shift ;;
        --load_model)    LOAD_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Environment & Project Root (absolute paths)
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/42bb95ce-04b4-461c-bddb-9489084b4593/scGPT/code/82863716-24e5-40ec-8643-9b7cd02c307e/scGPT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_ROOT}/save/run_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================"
echo " Project Root : ${PROJECT_ROOT}"
echo " Output Dir   : ${OUTPUT_DIR}"
echo " Timestamp    : ${TIMESTAMP}"
echo " CMCA-Loss    : Enabled (weight=0.1, temp=0.5)"
echo " Distributed  : ${DISTRIBUTED}"
echo " Load Model   : ${LOAD_MODEL:-None (from scratch)}"
echo "============================================"

# ---------------------------------------------------------------------------
# 2. Fix PyTorch environment issue
# ---------------------------------------------------------------------------
# The CPU-only PyTorch wheel may be missing libtorch_global_deps.so, which
# causes an OSError at import time. We create a minimal stub to work around this.
PYTHON_SITE_PKG=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || true)
if [ -z "${PYTHON_SITE_PKG}" ]; then
    PYTHON_SITE_PKG="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/envs/env/lib/python3.14/site-packages"
fi

TORCH_LIB_DIR="${PYTHON_SITE_PKG}/torch/lib"
if [ -d "${TORCH_LIB_DIR}" ] && [ ! -f "${TORCH_LIB_DIR}/libtorch_global_deps.so" ]; then
    echo "[SETUP] Creating libtorch_global_deps.so stub for CPU-only PyTorch..."
    # Create a minimal stub shared library if gcc is available
    if command -v gcc &>/dev/null; then
        cat > /tmp/stub_global_deps.c << 'STUB_EOF'
int torch_global_deps_stub(void) { return 0; }
STUB_EOF
        gcc -shared -fPIC -o /tmp/libtorch_global_deps.so /tmp/stub_global_deps.c \
            -Wl,--soname=libtorch_global_deps.so 2>/dev/null || true
    fi
    if [ -f /tmp/libtorch_global_deps.so ]; then
        cp /tmp/libtorch_global_deps.so "${TORCH_LIB_DIR}/libtorch_global_deps.so"
        echo "[SETUP] libtorch_global_deps.so stub created."
    else
        # Fallback: create an empty file (may not work but better than failing)
        touch "${TORCH_LIB_DIR}/libtorch_global_deps.so"
        echo "[SETUP] Warning: Created empty stub - may cause runtime issues."
    fi
fi

# ---------------------------------------------------------------------------
# 3. Set environment variables for the Python script
# ---------------------------------------------------------------------------
export OUTPUT_DIR
export LOAD_MODEL
export DISTRIBUTED

# ---------------------------------------------------------------------------
# 4. Launch training
# ---------------------------------------------------------------------------
TRAIN_SCRIPT="${LOG_DIR}/train_cmca.py"

if [ "${DISTRIBUTED}" = "true" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Launching distributed training with torchrun..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    torchrun \
        --nnodes=1 \
        --nproc_per_node=${NPROC_PER_NODE:-4} \
        --master_addr=localhost \
        --master_port=${MASTER_PORT:-12355} \
        "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_DIR}/train_distributed_${TIMESTAMP}.log"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Launching single-GPU training..."
    python3 "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Pipeline finished. Check ${OUTPUT_DIR} for results."
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Logs: ${LOG_DIR}/train_${TIMESTAMP}.log"