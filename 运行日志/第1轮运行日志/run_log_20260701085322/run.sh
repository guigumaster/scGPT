#!/bin/bash
# =============================================================================
# scGPT: 标记基因引导的自适应掩码策略 (Marker Gene Guided Adaptive Masking)
# 训练、验证、测试全流程脚本
# =============================================================================

set -e

# ===================== 环境配置 =====================
# 项目根目录
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/fc57bee6-9910-4552-a4bf-a4cd016ddc60/scGPT/code/585ae2b7-712a-4214-b6d9-47a3934a2167/scGPT"
cd "${PROJECT_ROOT}"

# 日志和输出目录
RUN_LOG_DIR="${PROJECT_ROOT}/run_log"
mkdir -p "${RUN_LOG_DIR}"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RUN_LOG_DIR}/training_${TIMESTAMP}.log"
echo "===== scGPT Training Pipeline Start at ${TIMESTAMP} =====" | tee -a "${LOG_FILE}"

# ===================== 环境检测 =====================
echo "===== 检测运行环境 =====" | tee -a "${LOG_FILE}"

# Python 环境
PYTHON_PATH=$(which python3 || which python)
echo "Python: ${PYTHON_PATH}" | tee -a "${LOG_FILE}"
python3 --version 2>&1 | tee -a "${LOG_FILE}"

# 检查 PyTorch 和 GPU
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# ===================== GPU 选择策略 =====================
echo "===== 检查 GPU 显存使用情况 =====" | tee -a "${LOG_FILE}"
nvidia-smi --format=csv --query-gpu=index,memory.used,memory.free,utilization.gpu 2>&1 | tee -a "${LOG_FILE}"

# 自动选择显存利用率最低的 GPU
echo "===== 选择最优 GPU =====" | tee -a "${LOG_FILE}"
FREE_GPU=$(python3 -c "
import subprocess, pandas as pd, sys, os
try:
    gpu_stats = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=index,memory.used,memory.free']).decode('utf-8')
    gpu_df = pd.read_csv(pd.io.common.StringIO(gpu_stats), names=['index', 'memory.used', 'memory.free'], skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].str.replace(' MiB', '').astype(int)
    # 选择显存空闲最多的 GPU
    best_idx = gpu_df['memory.free'].idxmax()
    best_gpu = int(gpu_df.iloc[best_idx]['index'])
    print(best_gpu)
except Exception as e:
    print(0)  # 默认使用 GPU 0
" 2>&1)
echo "Selected GPU: ${FREE_GPU}" | tee -a "${LOG_FILE}"
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"

# ===================== 数据准备 =====================
echo "===== 检查数据 =====" | tee -a "${LOG_FILE}"
DATA_FILE="${PROJECT_ROOT}/data/pbmc10k_scgpt_ready.h5ad"
if [ -f "${DATA_FILE}" ]; then
    echo "本地数据已存在: ${DATA_FILE}" | tee -a "${LOG_FILE}"
    python3 -c "
import scanpy as sc
adata = sc.read_h5ad('${DATA_FILE}')
print(f'数据维度: {adata.shape[0]} cells x {adata.shape[1]} genes')
print(f'细胞类型: {adata.obs[\"celltype\"].nunique()}')
print(f'批次信息: {adata.obs[\"batch\"].unique()}')
" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "本地数据不存在，将在训练脚本中自动下载 PBMC 10K 数据集" | tee -a "${LOG_FILE}"
fi

# ===================== 模型检查 =====================
echo "===== 检查预训练模型 =====" | tee -a "${LOG_FILE}"
PRETRAINED_DIR="${PROJECT_ROOT}/save/scGPT_human"
if [ -d "${PRETRAINED_DIR}" ]; then
    echo "预训练模型已存在: ${PRETRAINED_DIR}" | tee -a "${LOG_FILE}"
    ls -la "${PRETRAINED_DIR}/" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "警告: 预训练模型目录 ${PRETRAINED_DIR} 不存在！" | tee -a "${LOG_FILE}"
    echo "请先下载预训练模型到 ${PRETRAINED_DIR}" | tee -a "${LOG_FILE}"
    echo "下载地址: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing" | tee -a "${LOG_FILE}"
    echo "继续运行，将从头训练模型..." | tee -a "${LOG_FILE}"
fi

# ===================== 安装依赖 =====================
echo "===== 检查依赖 =====" | tee -a "${LOG_FILE}"
pip install -r "${PROJECT_ROOT}/requirements.txt" 2>&1 | tail -5 | tee -a "${LOG_FILE}"
pip install wandb 2>&1 | tail -3 | tee -a "${LOG_FILE}"

# ===================== 训练配置 =====================
echo "===== 配置训练参数 =====" | tee -a "${LOG_FILE}"

# 训练超参数 (标记基因引导自适应掩码策略)
EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-4
MASK_RATIO=0.4
MARKER_MASK_PROB=0.7       # 标记基因高掩码概率
NON_MARKER_MASK_PROB=0.25  # 非标记基因低掩码概率
N_MARKER_GENES=50          # 每种细胞类型的前 N 个标记基因
USE_MARKER_MASKING=True    # 启用标记基因引导掩码

echo "训练参数:" | tee -a "${LOG_FILE}"
echo "  Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "  Batch Size: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "  Learning Rate: ${LEARNING_RATE}" | tee -a "${LOG_FILE}"
echo "  Base Mask Ratio: ${MASK_RATIO}" | tee -a "${LOG_FILE}"
echo "  Marker Mask Prob: ${MARKER_MASK_PROB} (标记基因掩码概率)" | tee -a "${LOG_FILE}"
echo "  Non-Marker Mask Prob: ${NON_MARKER_MASK_PROB} (非标记基因掩码概率)" | tee -a "${LOG_FILE}"
echo "  N Marker Genes: ${N_MARKER_GENES} (每种细胞类型标记基因数)" | tee -a "${LOG_FILE}"
echo "  Use Marker Guided Masking: ${USE_MARKER_MASKING}" | tee -a "${LOG_FILE}"

# ===================== 训练模型 =====================
echo "===== 开始训练 =====" | tee -a "${LOG_FILE}"
echo "训练脚本: ${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" | tee -a "${LOG_FILE}"
echo "开始时间: $(date)" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# WANDB 模式设置 (离线模式避免网络问题)
export WANDB_MODE="offline"

python3 "${PROJECT_ROOT}/tutorials/Tutorial_Integration.py" 2>&1 | tee -a "${LOG_FILE}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
echo "训练结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "训练退出码: ${TRAIN_EXIT_CODE}" | tee -a "${LOG_FILE}"

# ===================== 查找最佳模型 =====================
echo "===== 查找最佳模型 =====" | tee -a "${LOG_FILE}"
BEST_MODEL_DIR=$(find "${PROJECT_ROOT}/save" -type d -name "dev_PBMC_10K*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "${BEST_MODEL_DIR}" ]; then
    echo "最新模型目录: ${BEST_MODEL_DIR}" | tee -a "${LOG_FILE}"
    ls -la "${BEST_MODEL_DIR}/" 2>&1 | tee -a "${LOG_FILE}"
    
    BEST_MODEL_FILE="${BEST_MODEL_DIR}/best_model.pt"
    if [ -f "${BEST_MODEL_FILE}" ]; then
        echo "最佳模型: ${BEST_MODEL_FILE}" | tee -a "${LOG_FILE}"
    fi
else
    echo "未找到模型输出目录" | tee -a "${LOG_FILE}"
fi

# ===================== 结果汇总 =====================
echo "===== 训练结果汇总 =====" | tee -a "${LOG_FILE}"
echo "项目根目录: ${PROJECT_ROOT}" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "训练配置: 标记基因引导自适应掩码" | tee -a "${LOG_FILE}"
echo "  - 标记基因掩码概率: ${MARKER_MASK_PROB}" | tee -a "${LOG_FILE}"
echo "  - 非标记基因掩码概率: ${NON_MARKER_MASK_PROB}" | tee -a "${LOG_FILE}"
echo "  - 每种细胞类型标记基因数: ${N_MARKER_GENES}" | tee -a "${LOG_FILE}"
echo "完成时间: $(date)" | tee -a "${LOG_FILE}"
echo "===== 训练流程结束 =====" | tee -a "${LOG_FILE}"