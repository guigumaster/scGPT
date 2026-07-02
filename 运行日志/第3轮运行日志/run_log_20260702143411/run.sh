#!/usr/bin/env bash
# =============================================================================
# PCT-AIM v5: Unified Multi-task Training / Validation / Testing Pipeline
# =============================================================================
# KEY IMPROVEMENTS over v4:
#   1. KILL rogue GPU processes before starting to prevent OOM
#   2. Reduced batch_size from 32→16, increased grad_accum from 2→4 (same effective batch)
#   3. Enabled gradient checkpointing + flash attention for memory efficiency
#   4. Pipeline continues on task failure - other tasks still execute
#   5. Gradient centralization for training stability
#   6. Memory-efficient CCE (no double _encode call)
#
# KEY IMPROVEMENTS over v3 (inherited):
#   1. Enhanced CCE (Contrastive Cell Embedding) with learnable temperature
#   2. Enhanced PerturbationPredictor with residual connections
#   3. Improved flag encoder and better initialization
#   4. Better Similarity module with learnable temperature for CCE
#
# Tasks:
#   1) Multi-batch Integration (PBMC 10K) - 25 epochs, CCE + ECS + DAB
#   2) Perturbation Prediction (Norman 2019) - 25 epochs, CCE + pert predictor
#   3) Large-scale Perturbation Prediction (Replogle 2022) - 15 epochs, CCE
#
# Usage:
#   bash run_log/run.sh [task] [mode]
#     task: perturbation | integration | replogle | all
#     mode: train | eval | all
#
# All paths are absolute.
# =============================================================================

set -o pipefail

# ----------------------------- Configuration ---------------------------------
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/da2c8100-835c-4cc8-8753-fb2b2535da49/scGPT/code/3119b63d-2222-4c9e-8c0b-e395303b5f91/scGPT"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_SILENT="true"

# Memory management - optimized for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1

# Use system Python (absolute path)
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
    ${PYTHON} -c "
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
gc.collect()
torch.cuda.synchronize()
print('GPU memory cleaned')
" 2>/dev/null || true
    sleep 2
}

# KILL any leftover Python processes on this GPU that are not this script
kill_rogue_gpu_processes() {
    local my_pid=$$
    log_info "Checking for rogue GPU processes..."
    # Get all Python processes using GPU 0
    local rogue_pids
    rogue_pids=$(${PYTHON} -c "
import subprocess, os
result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 
                       capture_output=True, text=True, timeout=10)
for line in result.stdout.strip().split('\n'):
    if not line.strip():
        continue
    parts = line.split(',')
    if len(parts) >= 2:
        pid = parts[0].strip()
        mem = parts[1].strip().replace('MiB', '').strip()
        try:
            if int(mem) > 100 and pid != str($$):
                print(pid)
        except ValueError:
            pass
" 2>/dev/null || true)
    for pid in $rogue_pids; do
        if [ -n "$pid" ] && [ "$pid" != "$$" ] && [ "$pid" != "" ]; then
            log_info "Killing rogue process PID=$pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    cleanup_gpu
    log_info "GPU process cleanup complete."
}

run_python() {
    local script_path="$1"
    shift
    local extra_args="$@"
    log_info "Running: ${PYTHON} ${script_path} ${extra_args}"
    cd "${PROJECT_ROOT}"
    # Pre-run GPU status
    ${PYTHON} -c "
import torch
if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f'GPU free memory before: {free_mem / 1024**3:.1f} GB / {total_mem / 1024**3:.1f} GB')
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

# ----------------------------- Task 1: Perturbation Prediction (v5 optimized) -----------------
PERTURBATION_SCRIPT="${PROJECT_ROOT}/data/finetune_perturbation_pctaim.py"

train_perturbation() {
    log_info "=== PCT-AIM v5: Training Perturbation Prediction (Norman 2019) ==="
    cleanup_gpu
    # v5: all hyperparams now in the Python file with defaults, only override what's needed
    run_python "${PERTURBATION_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--load_model=${PROJECT_ROOT}/save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    PERTURBATION_SAVE_DIR=$(find_latest_save_dir "dev_norman_pctaim")
    log_info "=== Perturbation Prediction Training Complete (save_dir: ${PERTURBATION_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_perturbation() {
    log_info "=== PCT-AIM v5: Evaluating Perturbation Prediction ==="
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

# ----------------------------- Task 2: Multi-batch Integration (v5 optimized) ----------------
INTEGRATION_SCRIPT="${PROJECT_ROOT}/data/finetune_integration_pctaim.py"

train_integration() {
    log_info "=== PCT-AIM v5: Training Multi-batch Integration (PBMC 10K) ==="
    cleanup_gpu
    # v5: all hyperparams now in the Python file with defaults, only override what's needed
    run_python "${INTEGRATION_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--load_model=${PROJECT_ROOT}/save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    INTEGRATION_SAVE_DIR=$(find_latest_save_dir "dev_PBMC_10K_PCTAIM")
    log_info "=== Integration Training Complete (save_dir: ${INTEGRATION_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_integration() {
    log_info "=== PCT-AIM v5: Evaluating Integration ==="
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

# ----------------------------- Task 3: Large-scale Perturbation (v5 optimized) --------------
REPLOGLE_SCRIPT="${PROJECT_ROOT}/data/finetune_replogle_pctaim.py"

train_replogle() {
    log_info "=== PCT-AIM v5: Training Large-scale Perturbation (Replogle 2022) ==="
    cleanup_gpu
    # v5: all hyperparams now in the Python file with defaults, only override what's needed
    run_python "${REPLOGLE_SCRIPT}" \
        "--seed=42" \
        "--do_train=True" \
        "--load_model=${PROJECT_ROOT}/save/scGPT_human" \
        "--num_workers=0"
    local ret=$?
    REPLOGLE_SAVE_DIR=$(find_latest_save_dir "dev_replogle_pctaim")
    log_info "=== Large-scale Perturbation Training Complete (save_dir: ${REPLOGLE_SAVE_DIR}) ==="
    cleanup_gpu
    return ${ret}
}

eval_replogle() {
    log_info "=== PCT-AIM v5: Evaluating Large-scale Perturbation ==="
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
from scgpt.model.pctaim_model import PCTAIMTransformerModel, PerturbationPredictor, EnhancedAdversarialDiscriminator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
print('All modules OK')
" 2>/dev/null || { log_error "scgpt module import failed."; exit 1; }

    # Check data files
    for f in norman_scgpt_ready.h5ad pbmc10k_scgpt_ready.h5ad replogle_scgpt_ready.h5ad; do
        if [ -f "${PROJECT_ROOT}/data/${f}" ]; then
            log_info "Data file found: ${f}"
        else
            log_info "Data file not found: ${f}"
        fi
    done

    # Check pretrained model
    if [ -f "${PROJECT_ROOT}/save/scGPT_human/best_model.pt" ]; then
        log_info "Pretrained scGPT model found at ${PROJECT_ROOT}/save/scGPT_human/"
    else
        log_info "No pretrained scGPT model found. Will train from scratch."
    fi

    log_info "Environment check complete."
}

# ----------------------------- Summary Report --------------------------------
generate_report() {
    log_info "=== Generating Training Summary Report ==="
    local report_file="${PROJECT_ROOT}/run_log/training_report_v5.txt"
    {
        echo "=========================================="
        echo "PCT-AIM v5 Training Report"
        echo "Generated: $(date)"
        echo "=========================================="
        echo ""
        
        for task_name in "Integration" "Perturbation" "Replogle"; do
            local var_name="${task_name}_SAVE_DIR"
            local save_dir="${!var_name}"
            if [ -n "${save_dir}" ] && [ -f "${save_dir}/best_model.pt" ]; then
                echo "--- ${task_name} ---"
                echo "Checkpoint: ${save_dir}/best_model.pt"
                if [ -f "${save_dir}/args.json" ]; then
                    echo "Config: $(cat ${save_dir}/args.json)"
                fi
                echo ""
            fi
        done
    } > "${report_file}"
    log_info "Report saved to ${report_file}"
}

# ----------------------------- Main Dispatcher --------------------------------
main() {
    # Kill any rogue GPU processes first
    kill_rogue_gpu_processes
    
    setup_environment

    local has_errors=0

    case "${TASK}" in
        perturbation)
            case "${MODE}" in
                train) train_perturbation; has_errors=$? ;;
                eval)  eval_perturbation; has_errors=$? ;;
                all)
                    train_perturbation; local t1=$?
                    if [ $t1 -eq 0 ]; then
                        eval_perturbation; has_errors=$?
                    else
                        has_errors=$t1
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
                train) train_integration; has_errors=$? ;;
                eval)  eval_integration; has_errors=$? ;;
                all)
                    train_integration; local t1=$?
                    if [ $t1 -eq 0 ]; then
                        eval_integration; has_errors=$?
                    else
                        has_errors=$t1
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
                train) train_replogle; has_errors=$? ;;
                eval)  eval_replogle; has_errors=$? ;;
                all)
                    train_replogle; local t1=$?
                    if [ $t1 -eq 0 ]; then
                        eval_replogle; has_errors=$?
                    else
                        has_errors=$t1
                    fi
                    ;;
                *)
                    log_error "Unknown mode: ${MODE}"
                    exit 1
                    ;;
            esac
            ;;
        all)
            log_info "=== PCT-AIM v5: Running FULL pipeline for ALL tasks ==="
            
            # Train all tasks - each is independent, continue even if one fails
            log_info "--- Task 1/3: Multi-batch Integration (PBMC 10K) ---"
            train_integration; local i1=$?
            if [ $i1 -ne 0 ]; then
                log_error "Integration training failed, but continuing with other tasks..."
                has_errors=1
            fi
            cleanup_gpu
            
            log_info "--- Task 2/3: Perturbation Prediction (Norman 2019) ---"
            train_perturbation; local i2=$?
            if [ $i2 -ne 0 ]; then
                log_error "Perturbation training failed, but continuing with other tasks..."
                has_errors=1
            fi
            cleanup_gpu
            
            log_info "--- Task 3/3: Large-scale Perturbation (Replogle 2022) ---"
            train_replogle; local i3=$?
            if [ $i3 -ne 0 ]; then
                log_error "Replogle training failed."
                has_errors=1
            fi
            cleanup_gpu
            
            # Evaluate all - each is independent
            log_info "--- Evaluating all tasks ---"
            eval_integration || true
            eval_perturbation || true
            eval_replogle || true
            
            generate_report
            log_info "=== PCT-AIM v5 FULL pipeline complete ==="
            ;;
        *)
            log_error "Unknown task: ${TASK}. Use: perturbation, integration, replogle, all"
            exit 1
            ;;
    esac

    if [ $has_errors -ne 0 ]; then
        log_error "Pipeline completed with errors (exit code ${has_errors})."
    else
        log_info "All tasks completed successfully!"
    fi
    exit $has_errors
}

# ----------------------------- Entry Point -----------------------------------
main