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
#   bash run.sh [task] [mode]
#     task: perturbation | integration | replogle | all
#     mode: train | eval | test | all
#
# Example:
#   bash run.sh perturbation train     # Train perturbation model
#   bash run.sh all all                # Full pipeline for all tasks
# =============================================================================

set -euo pipefail

# ----------------------------- Configuration ---------------------------------
# ** IMPORTANT: Change PROJECT_ROOT to your absolute project path **
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/da2c8100-835c-4cc8-8753-fb2b2535da49/scGPT/code/3119b63d-2222-4c9e-8c0b-e395303b5f91/scGPT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}

TASK=${1:-all}
MODE=${2:-all}

# ----------------------------- Helper Functions ------------------------------
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

run_python() {
    local script="$1"
    shift
    local extra_args="$@"
    log_info "Running: python ${script} ${extra_args}"
    cd "${PROJECT_ROOT}"
    python "${script}" ${extra_args}
    local ret=$?
    if [ $ret -ne 0 ]; then
        log_error "Command failed with exit code ${ret}: python ${script} ${extra_args}"
        exit $ret
    fi
    log_info "Completed: python ${script} ${extra_args}"
}

# ----------------------------- Task 1: Perturbation Prediction -----------------
train_perturbation() {
    log_info "=== PCT-AIM: Training Perturbation Prediction (Norman 2019) ==="
    run_python "data/finetune_perturbation_pctaim.py" \
        "--seed=${SEED}" \
        "--epochs=50" \
        "--batch_size=16" \
        "--layer_size=512" \
        "--nlayers=4" \
        "--nhead=8" \
        "--dropout=0.2" \
        "--lr=5e-5" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=4" \
        "--use_cosine_scheduler=True" \
        "--warmup_epochs=3" \
        "--early_stopping_patience=5" \
        "--fast_transformer=True" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--task_type=perturbation"
    log_info "=== Perturbation Prediction Training Complete ==="
}

eval_perturbation() {
    log_info "=== PCT-AIM: Evaluating Perturbation Prediction ==="
    # Find the latest model checkpoint
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_norman_pctaim"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No perturbation model checkpoint found. Train first."
        return 1
    fi
    log_info "Using checkpoint: ${save_dir}/best_model.pt"
    run_python "data/finetune_perturbation_pctaim.py" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Perturbation Evaluation Complete ==="
}

test_perturbation() {
    log_info "=== PCT-AIM: Testing Perturbation Prediction ==="
    # Compute perturbation metrics: Pearson correlation, MSE, MRE
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_norman_pctaim"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No perturbation model found."
        return 1
    fi
    log_info "Testing with model: ${save_dir}/best_model.pt"
    # Run evaluation + compute scib metrics
    run_python "${PROJECT_ROOT}/scgpt/trainer.py" \
        "--task=perturb" \
        "--load_model=${save_dir}/best_model.pt"
    log_info "=== Perturbation Testing Complete ==="
}

# ----------------------------- Task 2: Multi-batch Integration ----------------
train_integration() {
    log_info "=== PCT-AIM: Training Multi-batch Integration (PBMC 10K) ==="
    run_python "data/finetune_integration_pctaim.py" \
        "--seed=${SEED}" \
        "--epochs=100" \
        "--batch_size=16" \
        "--layer_size=128" \
        "--nlayers=4" \
        "--nhead=4" \
        "--dropout=0.2" \
        "--lr=1e-4" \
        "--mask_ratio=0.4" \
        "--gradient_accumulation_steps=4" \
        "--ecs_thres=0.8" \
        "--dab_weight=1.0" \
        "--fast_transformer=True" \
        "--amp=True" \
        "--use_cross_modal=False" \
        "--task_type=integration"
    log_info "=== Integration Training Complete ==="
}

eval_integration() {
    log_info "=== PCT-AIM: Evaluating Integration ==="
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_PBMC_10K_PCTAIM"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No integration model checkpoint found. Train first."
        return 1
    fi
    log_info "Using checkpoint: ${save_dir}/best_model.pt"
    # Compute scib metrics (NMI, ARI, ASW, PCR_batch, graph_conn)
    run_python "${PROJECT_ROOT}/scgpt/utils/util.py" \
        "--eval_integration" \
        "--model_path=${save_dir}/best_model.pt"
    log_info "=== Integration Evaluation Complete ==="
}

test_integration() {
    log_info "=== PCT-AIM: Testing Integration ==="
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_PBMC_10K_PCTAIM"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No integration model found."
        return 1
    fi
    log_info "Testing with model: ${save_dir}/best_model.pt"
    # Generate UMAP visualizations and compute all scib metrics
    run_python "${PROJECT_ROOT}/data/finetune_integration_pctaim.py" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Integration Testing Complete ==="
}

# ----------------------------- Task 3: Large-scale Perturbation --------------
train_replogle() {
    log_info "=== PCT-AIM: Training Large-scale Perturbation (Replogle 2022) ==="
    run_python "data/finetune_replogle_pctaim.py" \
        "--seed=${SEED}" \
        "--epochs=80" \
        "--batch_size=8" \
        "--layer_size=256" \
        "--nlayers=4" \
        "--nhead=4" \
        "--dropout=0.3" \
        "--lr=1e-4" \
        "--mask_ratio=0.35" \
        "--gradient_accumulation_steps=8" \
        "--warmup_epochs=5" \
        "--early_stopping_patience=8" \
        "--dab_weight=0.3" \
        "--fast_transformer=True" \
        "--amp=True" \
        "--use_pert_cond=True" \
        "--pert_per_gene=False" \
        "--task_type=large_perturbation"
    log_info "=== Large-scale Perturbation Training Complete ==="
}

eval_replogle() {
    log_info "=== PCT-AIM: Evaluating Large-scale Perturbation ==="
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_replogle_pctaim"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No Replogle model checkpoint found. Train first."
        return 1
    fi
    log_info "Using checkpoint: ${save_dir}/best_model.pt"
    run_python "data/finetune_replogle_pctaim.py" \
        "--do_train=False" \
        "--load_model=${save_dir}"
    log_info "=== Replogle Evaluation Complete ==="
}

test_replogle() {
    log_info "=== PCT-AIM: Testing Large-scale Perturbation ==="
    local save_dir=$(ls -td "${PROJECT_ROOT}/save/dev_replogle_pctaim"* 2>/dev/null | head -1)
    if [ -z "${save_dir}" ]; then
        log_error "No Replogle model found."
        return 1
    fi
    log_info "Testing with model: ${save_dir}/best_model.pt"
    run_python "${PROJECT_ROOT}/scgpt/trainer.py" \
        "--task=perturb" \
        "--load_model=${save_dir}/best_model.pt"
    log_info "=== Replogle Testing Complete ==="
}

# ----------------------------- Environment Setup ------------------------------
setup_environment() {
    log_info "Setting up environment..."
    cd "${PROJECT_ROOT}"

    # Install dependencies if needed
    if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
        log_info "Creating virtual environment..."
        python -m venv "${PROJECT_ROOT}/.venv"
    fi
    source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null || true

    # Check critical packages
    log_info "Checking Python packages..."
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
        log_error "PyTorch not found. Please install requirements: pip install -r requirements.txt"
        exit 1
    }
    python -c "import wandb; print(f'wandb {wandb.__version__}')" 2>/dev/null || {
        log_info "Installing wandb..."
        pip install wandb -q
    }
    python -c "import scipy; print(f'scipy {scipy.__version__}')" 2>/dev/null || {
        pip install scipy -q
    }
    python -c "import scib; print(f'scib imported')" 2>/dev/null || {
        log_info "Installing scib..."
        pip install scib -q
    }

    # Check data files
    log_info "Checking data files..."
    local norman_data="${PROJECT_ROOT}/data/norman_scgpt_ready.h5ad"
    local replogle_data="${PROJECT_ROOT}/data/replogle_scgpt_ready.h5ad"

    if [ ! -f "${norman_data}" ]; then
        log_info "Norman data not found locally. Will be downloaded during training."
    fi
    if [ ! -f "${replogle_data}" ]; then
        log_info "Replogle data not found locally. Will be downloaded during training."
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
            train_integration
            train_replogle
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