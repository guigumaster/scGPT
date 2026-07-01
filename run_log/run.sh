#!/usr/bin/env bash
# =============================================================================
# PCT-AIM: Unified Training / Validation / Testing Pipeline
# =============================================================================
# This script runs the full PCT-AIM pipeline for all three tasks:
#   1) Perturbation Prediction (Norman 2019)
#   2) Multi-batch Integration (PBMC 10K)
#   3) Large-scale Perturbation Prediction (Replogle 2022)
#
# Usage:
#   bash run_log/run.sh [task] [mode]
#     task: perturbation | integration | replogle | all
#     mode: train | eval | test | all
#
# All paths are absolute.
# =============================================================================

set -o pipefail

# ----------------------------- Configuration ---------------------------------
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/da2c8100-835c-4cc8-8753-fb2b2535da49/scGPT/code/3119b63d-2222-4c9e-8c0b-e395303b5f91/scGPT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_SILENT="true"

# Memory management: reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Threading optimizations to avoid multi-threaded fork issues
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Function to clear GPU memory between tasks
cleanup_gpu() {
    log_info "Cleaning GPU memory..."
    ${PYTHON} -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect(); gc.collect(); print('GPU memory cleaned')" 2>/dev/null || true
    sleep 5
}

# Use system Python
PYTHON="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python3"

cd "${PROJECT_ROOT}"

NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-42}

TASK=${1:-all}
MODE=${2:-all}

# Track save directories for eval
PERTURBATION_SAVE_DIR=""
INTEGRATION_SAVE_DIR=""
REPLOGLE_SAVE_DIR=""

# ----------------------------- Helper Functions ------------------------------
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

run_python() {
    local script_path="$1"
    shift
    local extra_args="$@"
    log_info "Running: ${PYTHON} ${script_path} ${extra_args}"
    cd "${PROJECT_ROOT}"
    # Check GPU memory before running
    ${PYTHON} -c "
import torch
if torch.cuda.is_available():
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f'GPU free memory before: {free_mem:.1f} GB')
    torch.cuda.empty_cache()
import gc; gc.collect()
" 2>/dev/null || true
    ${PYTHON} "${script_path}" ${extra_args}
    local ret=$?
    if [ $ret -ne 0 ]; then
        log_error "Command failed with exit code ${ret}: ${PYTHON} ${script_path} ${extra_args}"
        exit $ret
    fi
    log_info "Completed: ${PYTHON} ${script_path} ${extra_args}"
}

# Find the latest save directory for a given pattern
find_latest_save_dir() {
    local pattern="$1"
    local dir=$(ls -td "${PROJECT_ROOT}/save/${pattern}"* 2>/dev/null | head -1)
    echo "${dir}"
}

# Check if a save directory has the required files for evaluation
check_save_dir() {
    local save_dir="$1"
    local name="$2"
    if [ -z "${save_dir}" ]; then
        log_error "No ${name} model checkpoint found. Train first."
        return 1
    fi
    if [ ! -f "${save_dir}/best_model.pt" ]; then
        log_error "${name} checkpoint missing best_model.pt in ${save_dir}."
        return 1
    fi
    if [ ! -f "${save_dir}/vocab.json" ]; then
        log_error "${name} checkpoint missing vocab.json in ${save_dir}."
        log_info "The vocab.json should have been saved during training."
        log_info "Try re-running training first."
        return 1
    fi
    log_info "Found ${name} checkpoint: ${save_dir}/best_model.pt"
    return 0
}

# ----------------------------- Task 1: Perturbation Prediction -----------------
PERTURBATION_SCRIPT="${PROJECT_ROOT}/data/finetune_perturbation_pctaim.py"

train_perturbation() {
    log_info "=== PCT-AIM: Training Perturbation Prediction (Norman 2019) ==="
    cleanup_gpu
    run_python "${PERTURBATION_SCRIPT}" \
        "--seed=${SEED}" \
        "--epochs=10" \
        "--batch_size=8" \
        "--layer_size=256" \
        "--nlayers=4" \
        "--nhead=8" \
        "--dropout=0.2" \
        "--lr=5e-5" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=4" \
        "--use_cosine_scheduler=True" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=5" \
        "--fast_transformer=False" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--pert_per_gene=False" \
        "--task_type=perturbation" \
        "--num_workers=0"
    # Store the latest save dir for later eval
    PERTURBATION_SAVE_DIR=$(find_latest_save_dir "dev_norman_pctaim")
    log_info "=== Perturbation Prediction Training Complete (save_dir: ${PERTURBATION_SAVE_DIR}) ==="
    cleanup_gpu
}

eval_perturbation() {
    log_info "=== PCT-AIM: Evaluating Perturbation Prediction ==="
    local save_dir="${PERTURBATION_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_norman_pctaim")
    fi
    check_save_dir "${save_dir}" "Perturbation" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${PERTURBATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Perturbation Evaluation Complete ==="
}

test_perturbation() {
    log_info "=== PCT-AIM: Testing Perturbation Prediction ==="
    local save_dir="${PERTURBATION_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_norman_pctaim")
    fi
    check_save_dir "${save_dir}" "Perturbation" || return 1
    log_info "Testing with model: ${save_dir}"
    run_python "${PERTURBATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Perturbation Testing Complete ==="
}

# ----------------------------- Task 2: Multi-batch Integration ----------------
INTEGRATION_SCRIPT="${PROJECT_ROOT}/data/finetune_integration_pctaim.py"

train_integration() {
    log_info "=== PCT-AIM: Training Multi-batch Integration (PBMC 10K) ==="
    cleanup_gpu
    run_python "${INTEGRATION_SCRIPT}" \
        "--seed=${SEED}" \
        "--epochs=10" \
        "--batch_size=8" \
        "--layer_size=128" \
        "--nlayers=4" \
        "--nhead=4" \
        "--dropout=0.2" \
        "--lr=1e-4" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=4" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=8" \
        "--ecs_thres=0.8" \
        "--dab_weight=1.0" \
        "--fast_transformer=False" \
        "--amp=True" \
        "--use_cross_modal=False" \
        "--num_workers=0"
    # Store the latest save dir for later eval
    INTEGRATION_SAVE_DIR=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    log_info "=== Integration Training Complete (save_dir: ${INTEGRATION_SAVE_DIR}) ==="
    cleanup_gpu
}

eval_integration() {
    log_info "=== PCT-AIM: Evaluating Integration ==="
    local save_dir="${INTEGRATION_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    fi
    check_save_dir "${save_dir}" "Integration" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${INTEGRATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Integration Evaluation Complete ==="
}

test_integration() {
    log_info "=== PCT-AIM: Testing Integration ==="
    local save_dir="${INTEGRATION_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    fi
    check_save_dir "${save_dir}" "Integration" || return 1
    log_info "Testing with model: ${save_dir}/best_model.pt"
    run_python "${INTEGRATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Integration Testing Complete ==="
}

# ----------------------------- Task 3: Large-scale Perturbation --------------
REPLOGLE_SCRIPT="${PROJECT_ROOT}/data/finetune_replogle_pctaim.py"

train_replogle() {
    log_info "=== PCT-AIM: Training Large-scale Perturbation (Replogle 2022) ==="
    cleanup_gpu
    run_python "${REPLOGLE_SCRIPT}" \
        "--seed=${SEED}" \
        "--epochs=10" \
        "--batch_size=8" \
        "--layer_size=256" \
        "--nlayers=4" \
        "--nhead=4" \
        "--dropout=0.3" \
        "--lr=1e-4" \
        "--mask_ratio=0.35" \
        "--gradient_accumulation_steps=4" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=6" \
        "--dab_weight=0.3" \
        "--fast_transformer=True" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--pert_per_gene=False" \
        "--task_type=large_perturbation" \
        "--num_workers=0"
    # Store the latest save dir for later eval
    REPLOGLE_SAVE_DIR=$(find_latest_save_dir "dev_replogle_pctaim")
    log_info "=== Large-scale Perturbation Training Complete (save_dir: ${REPLOGLE_SAVE_DIR}) ==="
    cleanup_gpu
}

eval_replogle() {
    log_info "=== PCT-AIM: Evaluating Large-scale Perturbation ==="
    local save_dir="${REPLOGLE_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_replogle_pctaim")
    fi
    check_save_dir "${save_dir}" "Replogle" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${REPLOGLE_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Replogle Evaluation Complete ==="
}

test_replogle() {
    log_info "=== PCT-AIM: Testing Large-scale Perturbation ==="
    local save_dir="${REPLOGLE_SAVE_DIR}"
    if [ -z "${save_dir}" ]; then
        save_dir=$(find_latest_save_dir "dev_replogle_pctaim")
    fi
    check_save_dir "${save_dir}" "Replogle" || return 1
    log_info "Testing with model: ${save_dir}"
    run_python "${REPLOGLE_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Replogle Testing Complete ==="
}

# ----------------------------- Environment Setup ------------------------------
setup_environment() {
    log_info "Setting up environment..."
    cd "${PROJECT_ROOT}"

    # Check critical packages
    log_info "Checking Python packages..."
    ${PYTHON} -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
" 2>/dev/null || {
        log_error "PyTorch not found. Please check system Python installation."
        exit 1
    }

    # Verify scgpt module importable
    ${PYTHON} -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from scgpt.model.pctaim_model import PCTAIMTransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
print('All modules OK')
" 2>/dev/null || {
        log_error "scgpt module import failed. Check PYTHONPATH."
        exit 1
    }

    # Check data files
    log_info "Checking data files..."
    local norman_data="${PROJECT_ROOT}/data/norman_scgpt_ready.h5ad"
    local pbmc_data="${PROJECT_ROOT}/data/pbmc10k_scgpt_ready.h5ad"
    local replogle_data="${PROJECT_ROOT}/data/replogle_scgpt_ready.h5ad"

    if [ -f "${norman_data}" ]; then
        log_info "Norman data found: ${norman_data}"
    else
        log_error "Norman data not found at ${norman_data}."
        exit 1
    fi
    if [ -f "${pbmc_data}" ]; then
        log_info "PBMC data found: ${pbmc_data}"
    else
        log_info "PBMC data not found at ${pbmc_data}. Will use scvi.data.pbmc_dataset()."
    fi
    if [ -f "${replogle_data}" ]; then
        log_info "Replogle data found: ${replogle_data}"
    else
        log_error "Replogle data not found at ${replogle_data}."
        exit 1
    fi

    # Check pretrained model
    if [ -f "${PROJECT_ROOT}/save/scGPT_human/best_model.pt" ]; then
        log_info "Pretrained scGPT model found at ${PROJECT_ROOT}/save/scGPT_human/"
    else
        log_info "No pretrained scGPT model found. Will train from scratch."
    fi

    log_info "Environment check complete."
}

# ----------------------------- Main Dispatcher --------------------------------
main() {
    setup_environment

    case "${TASK}" in
        perturbation)
            case "${MODE}" in
                train) train_perturbation ;;
                eval)  eval_perturbation ;;
                test)  test_perturbation ;;
                all)
                    train_perturbation
                    eval_perturbation
                    test_perturbation
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}. Use: train, eval, test, all"
                    exit 1
                    ;;
            esac
            ;;
        integration)
            case "${MODE}" in
                train) train_integration ;;
                eval)  eval_integration ;;
                test)  test_integration ;;
                all)
                    train_integration
                    eval_integration
                    test_integration
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}"
                    exit 1
                    ;;
            esac
            ;;
        replogle)
            case "${MODE}" in
                train) train_replogle ;;
                eval)  eval_replogle ;;
                test)  test_replogle ;;
                all)
                    train_replogle
                    eval_replogle
                    test_replogle
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}"
                    exit 1
                    ;;
            esac
            ;;
        all)
            log_info "=== PCT-AIM: Running FULL pipeline for ALL tasks ==="
            # Train all models
            train_perturbation
            cleanup_gpu
            train_integration
            cleanup_gpu
            train_replogle
            cleanup_gpu
            # Evaluate all models
            eval_perturbation
            eval_integration
            eval_replogle
            # Test all models
            test_perturbation
            test_integration
            test_replogle
            log_info "=== PCT-AIM FULL pipeline complete ==="
            ;;
        *)
            log_error "Unknown task: ${TASK}. Use: perturbation, integration, replogle, all"
            exit 1
            ;;
    esac

    log_info "All tasks completed successfully!"
}

# ----------------------------- Entry Point -----------------------------------
main