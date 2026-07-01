#!/usr/bin/env bash
# =============================================================================
# PCT-AIM v2: Unified Training / Validation / Testing Pipeline
# =============================================================================
# This script runs the full PCT-AIM v2 pipeline for all three tasks:
#   1) Perturbation Prediction (Norman 2019) - 30 epochs, 512d, pert predictor
#   2) Multi-batch Integration (PBMC 10K) - 30 epochs, 512d, pretrained
#   3) Large-scale Perturbation Prediction (Replogle 2022) - 20 epochs, 512d
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

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Use system Python
PYTHON="/inspire/cpfs/project/sais-ai-for-science-code/public/conda/miniconda3/bin/python3"

cd "${PROJECT_ROOT}"

TASK=${1:-all}
MODE=${2:-all}

# Track save directories
PERTURBATION_SAVE_DIR=""
INTEGRATION_SAVE_DIR=""
REPLOGLE_SAVE_DIR=""

# ----------------------------- Helper Functions ------------------------------
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

cleanup_gpu() {
    log_info "Cleaning GPU memory..."
    ${PYTHON} -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect(); gc.collect(); print('GPU memory cleaned')" 2>/dev/null || true
    sleep 5
}

run_python() {
    local script_path="$1"
    shift
    local extra_args="$@"
    log_info "Running: ${PYTHON} ${script_path} ${extra_args}"
    cd "${PROJECT_ROOT}"
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
    fi
    log_info "Completed: ${PYTHON} ${script_path} ${extra_args}"
    return ${ret}
}

find_latest_save_dir() {
    local pattern="$1"
    local dir=$(ls -td "${PROJECT_ROOT}/save/${pattern}"* 2>/dev/null | head -1)
    echo "${dir}"
}

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
        return 1
    fi
    log_info "Found ${name} checkpoint: ${save_dir}/best_model.pt"
    return 0
}

# ----------------------------- Task 1: Perturbation Prediction -----------------
PERTURBATION_SCRIPT="${PROJECT_ROOT}/data/finetune_perturbation_pctaim.py"

train_perturbation() {
    log_info "=== PCT-AIM v2: Training Perturbation Prediction (Norman 2019) ==="
    cleanup_gpu
    run_python "${PERTURBATION_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--epochs=30" \
        "--batch_size=24" \
        "--layer_size=512" \
        "--nlayers=12" \
        "--nhead=8" \
        "--d_hid=512" \
        "--dropout=0.15" \
        "--lr=5e-5" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=2" \
        "--use_cosine_scheduler=True" \
        "--warmup_epochs=5" \
        "--early_stopping_patience=10" \
        "--fast_transformer=False" \
        "--pre_norm=True" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--pert_per_gene=False" \
        "--task_type=perturbation" \
        "--load_model=save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    PERTURBATION_SAVE_DIR=$(find_latest_save_dir "dev_norman_pctaim")
    log_info "=== Perturbation Prediction Training Complete (save_dir: ${PERTURBATION_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_perturbation() {
    log_info "=== PCT-AIM v2: Evaluating Perturbation Prediction ==="
    local save_dir="${PERTURBATION_SAVE_DIR}"
    [ -z "${save_dir}" ] && save_dir=$(find_latest_save_dir "dev_norman_pctaim")
    check_save_dir "${save_dir}" "Perturbation" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${PERTURBATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    local ret=$?
    log_info "=== Perturbation Evaluation Complete ==="
    return ${ret}
}

# ----------------------------- Task 2: Multi-batch Integration ----------------
INTEGRATION_SCRIPT="${PROJECT_ROOT}/data/finetune_integration_pctaim.py"

train_integration() {
    log_info "=== PCT-AIM v2: Training Multi-batch Integration (PBMC 10K) ==="
    cleanup_gpu
    run_python "${INTEGRATION_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--epochs=30" \
        "--batch_size=24" \
        "--layer_size=512" \
        "--nlayers=12" \
        "--nhead=8" \
        "--d_hid=512" \
        "--dropout=0.15" \
        "--lr=1e-4" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=2" \
        "--use_cosine_scheduler=True" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=10" \
        "--ecs_thres=0.8" \
        "--dab_weight=1.0" \
        "--fast_transformer=False" \
        "--pre_norm=True" \
        "--amp=True" \
        "--use_pert_cond=False" \
        "--use_cross_modal=False" \
        "--task_type=integration" \
        "--load_model=save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    INTEGRATION_SAVE_DIR=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    log_info "=== Integration Training Complete (save_dir: ${INTEGRATION_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_integration() {
    log_info "=== PCT-AIM v2: Evaluating Integration ==="
    local save_dir="${INTEGRATION_SAVE_DIR}"
    [ -z "${save_dir}" ] && save_dir=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    check_save_dir "${save_dir}" "Integration" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${INTEGRATION_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    local ret=$?
    log_info "=== Integration Evaluation Complete ==="
    return ${ret}
}

# ----------------------------- Task 3: Large-scale Perturbation --------------
REPLOGLE_SCRIPT="${PROJECT_ROOT}/data/finetune_replogle_pctaim.py"

train_replogle() {
    log_info "=== PCT-AIM v2: Training Large-scale Perturbation (Replogle 2022) ==="
    cleanup_gpu
    run_python "${REPLOGLE_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--epochs=20" \
        "--batch_size=24" \
        "--layer_size=512" \
        "--nlayers=12" \
        "--nhead=8" \
        "--d_hid=512" \
        "--dropout=0.2" \
        "--lr=5e-5" \
        "--mask_ratio=0.35" \
        "--gradient_accumulation_steps=2" \
        "--use_cosine_scheduler=True" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=8" \
        "--dab_weight=0.3" \
        "--fast_transformer=False" \
        "--pre_norm=True" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--pert_per_gene=False" \
        "--task_type=large_perturbation" \
        "--load_model=save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    REPLOGLE_SAVE_DIR=$(find_latest_save_dir "dev_replogle_pctaim")
    log_info "=== Large-scale Perturbation Training Complete (save_dir: ${REPLOGLE_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_replogle() {
    log_info "=== PCT-AIM v2: Evaluating Large-scale Perturbation ==="
    local save_dir="${REPLOGLE_SAVE_DIR}"
    [ -z "${save_dir}" ] && save_dir=$(find_latest_save_dir "dev_replogle_pctaim")
    check_save_dir "${save_dir}" "Replogle" || return 1
    log_info "Using checkpoint: ${save_dir}"
    run_python "${REPLOGLE_SCRIPT}" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    local ret=$?
    log_info "=== Replogle Evaluation Complete ==="
    return ${ret}
}

# ----------------------------- Environment Setup ------------------------------
setup_environment() {
    log_info "Setting up environment..."
    cd "${PROJECT_ROOT}"

    log_info "Checking Python packages..."
    ${PYTHON} -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
" 2>/dev/null || { log_error "PyTorch not found."; exit 1; }

    ${PYTHON} -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from scgpt.model.pctaim_model import PCTAIMTransformerModel, PerturbationPredictor
from scgpt.tokenizer.gene_tokenizer import GeneVocab
print('All modules OK')
" 2>/dev/null || { log_error "scgpt module import failed."; exit 1; }

    # Check data files
    for f in norman_scgpt_ready.h5ad pbmc10k_scgpt_ready.h5ad replogle_scgpt_ready.h5ad; do
        if [ -f "${PROJECT_ROOT}/data/${f}" ]; then
            log_info "Data file found: ${f}"
        else
            log_info "Data file not found: ${f} (may be generated at runtime)"
        fi
    done

    # Check pretrained model
    if [ -f "${PROJECT_ROOT}/save/scGPT_human/best_model.pt" ]; then
        log_info "Pretrained scGPT model found at ${PROJECT_ROOT}/save/scGPT_human/"
    else
        log_info "No pretrained scGPT model found. Will train from scratch (may reduce performance)."
    fi

    log_info "Environment check complete."
}

# ----------------------------- Main Dispatcher --------------------------------
main() {
    setup_environment

    local overall_exit=0

    case "${TASK}" in
        perturbation)
            case "${MODE}" in
                train) train_perturbation; overall_exit=$? ;;
                eval)  eval_perturbation; overall_exit=$? ;;
                all)
                    train_perturbation; overall_exit=$?
                    if [ $overall_exit -eq 0 ]; then
                        eval_perturbation; overall_exit=$?
                    fi
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}. Use: train, eval, all"
                    exit 1
                    ;;
            esac
            ;;
        integration)
            case "${MODE}" in
                train) train_integration; overall_exit=$? ;;
                eval)  eval_integration; overall_exit=$? ;;
                all)
                    train_integration; overall_exit=$?
                    if [ $overall_exit -eq 0 ]; then
                        eval_integration; overall_exit=$?
                    fi
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}"
                    exit 1
                    ;;
            esac
            ;;
        replogle)
            case "${MODE}" in
                train) train_replogle; overall_exit=$? ;;
                eval)  eval_replogle; overall_exit=$? ;;
                all)
                    train_replogle; overall_exit=$?
                    if [ $overall_exit -eq 0 ]; then
                        eval_replogle; overall_exit=$?
                    fi
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}"
                    exit 1
                    ;;
            esac
            ;;
        all)
            log_info "=== PCT-AIM v2: Running FULL pipeline for ALL tasks ==="
            train_perturbation; overall_exit=$?
            cleanup_gpu
            if [ $overall_exit -eq 0 ]; then
                train_integration; overall_exit=$?
                cleanup_gpu
            fi
            if [ $overall_exit -eq 0 ]; then
                train_replogle; overall_exit=$?
                cleanup_gpu
            fi
            if [ $overall_exit -eq 0 ]; then
                eval_perturbation; overall_exit=$?
            fi
            if [ $overall_exit -eq 0 ]; then
                eval_integration; overall_exit=$?
            fi
            if [ $overall_exit -eq 0 ]; then
                eval_replogle; overall_exit=$?
            fi
            log_info "=== PCT-AIM v2 FULL pipeline complete ==="
            ;;
        *)
            log_error "Unknown task: ${TASK}. Use: perturbation, integration, replogle, all"
            exit 1
            ;;
    esac

    if [ $overall_exit -ne 0 ]; then
        log_error "Pipeline completed with errors (exit code ${overall_exit})."
    else
        log_info "All tasks completed successfully!"
    fi
    exit $overall_exit
}

# ----------------------------- Entry Point -----------------------------------
main