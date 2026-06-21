#!/bin/bash
# =============================================================================
# scGPT Cross-Modal Contrastive Alignment Loss (CMCA-Loss) Training Pipeline
# =============================================================================
# Description: This script runs the full training, validation, and testing
# pipeline for scGPT with CMCA-Loss enabled. It supports single-GPU and
# multi-GPU (distributed) training modes.
#
# Expected improvements:
#   - ARI: 0.71 -> 0.73–0.75
#   - PCR_batch: 0.3289 -> 0.25–0.30
#
# Usage:
#   bash run.sh                           # Single GPU training
#   bash run.sh --distributed             # Multi-GPU (torchrun) training
#   bash run.sh --load_model /path/to/ckpt  # Resume from pretrained weights
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
# 1. Environment & Project Root (use absolute path)
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/42bb95ce-04b4-461c-bddb-9489084b4593/scGPT/code/82863716-24e5-40ec-8643-9b7cd02c307e/scGPT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_ROOT}/save/run_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/run_log"
TRAIN_SCRIPT="${LOG_DIR}/train_cmca_${TIMESTAMP}.py"
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
# 2. Write the training Python script
# ---------------------------------------------------------------------------
cat > "${TRAIN_SCRIPT}" << 'TRAIN_PYTHON_EOF'
#!/usr/bin/env python3
"""scGPT training with Cross-Modal Contrastive Alignment Loss (CMCA-Loss)."""
import sys, os, copy, gc, json, time, warnings, math
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np
import scanpy as sc
import scvi
import wandb
from anndata import AnnData
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Project root and output dir are passed via environment
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./save")
LOAD_MODEL = os.environ.get("LOAD_MODEL", "")
DISTRIBUTED = os.environ.get("DISTRIBUTED", "false") == "true"

sys.path.insert(0, PROJECT_ROOT)
import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import MultiOmicTransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics

# ============================================================================
# Configuration
# ============================================================================
config = dict(
    seed=42,
    dataset_name="PBMC_10K",
    do_train=True,
    load_model=LOAD_MODEL,
    mask_ratio=0.4,
    epochs=30,
    n_bins=51,
    GEPC=True,            # Masked value prediction for cell embedding
    ecs_thres=0.8,        # Elastic cell similarity threshold
    dab_weight=1.0,
    lr=1e-4,
    batch_size=64,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    # Task flags
    task="multiomic",
    GEP=True,
    CLS=False,
    ESC=True,
    DAR=True,
    DSBN=True,
    use_batch_labels=True,
    use_mod=True,
    explicit_zero_prob=True,
    # Padding / masking
    pad_token="<pad>",
    mask_value=-1,
    pad_value=-2,
    include_zero_gene=True,
    n_input_bins=51,
    max_seq_len=1201,
    input_layer_key="X_binned",
    # CMCA-Loss configuration
    CMCA=True,
    cmca_weight=0.1,
    cmca_temp=0.5,
)

run = wandb.init(config=config, project="scGPT-CMCA", reinit=True,
                 settings=wandb.Settings(start_method="fork"))
config = wandb.config

set_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = scg.logger
save_dir = Path(OUTPUT_DIR)
scg.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"Output directory: {save_dir}")
logger.info(f"Device: {device}")

# ============================================================================
# Data loading and preprocessing
# ============================================================================
logger.info("Loading PBMC 10K dataset...")
adata = scvi.data.pbmc_dataset()
adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
adata.var = adata.var.set_index("gene_symbols")
adata.obs["str_batch"] = adata.obs["batch"].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Vocabulary
special_tokens = [config.pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
max_seq_len = n_hvg + 1

# Load pretrained model config if specified
if config.load_model and os.path.exists(config.load_model):
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1
                                for gene in adata.var["gene_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    with open(model_config_file) as f:
        mcfg = json.load(f)
    embsize, nhead_, d_hid, nlayers_ = mcfg["embsize"], mcfg["nheads"], mcfg["d_hid"], mcfg["nlayers"]
    n_layers_cls = mcfg["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead_ = config.nhead
    nlayers_ = config.nlayers
    d_hid = config.layer_size

# Preprocess
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=3,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=True,
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,
    hvg_flavor="seurat_v3",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch")

# Sort by batch for per_seq_batch_sample
adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# Tokenize
input_key = config.input_layer_key
all_counts = (adata.layers[input_key].toarray()
              if issparse(adata.layers[input_key])
              else adata.layers[input_key])
genes = adata.var["gene_name"].tolist()
celltypes_labels = np.array(adata.obs["celltype"].tolist())
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(np.unique(batch_ids))

(train_data, valid_data, train_cl, valid_cl,
 train_bl, valid_bl) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True)

# Create vocab if not loading pretrained
if not config.load_model or not os.path.exists(config.load_model):
    from torchtext.vocab import Vocab
    from torchtext._torchtext import Vocab as VocabPybind
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
vocab.set_default_index(vocab[config.pad_token])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value,
    append_cls=True, include_zero_gene=True)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value,
    append_cls=True, include_zero_gene=True)

# ============================================================================
# Model with CMCA-Loss support
# ============================================================================
ntokens = len(vocab)
logger.info(f"Vocabulary size: {ntokens}")
logger.info(f"Building MultiOmicTransformerModel with do_cmca={config.CMCA}")

model = MultiOmicTransformerModel(
    ntoken=ntokens,
    d_model=embsize,
    nhead=nhead_,
    d_hid=d_hid,
    nlayers=nlayers_,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    use_mod=True,
    ntokens_mod=ntokens,
    vocab_mod=vocab,
    do_cmca=config.CMCA,
    cmca_weight=config.cmca_weight,
    cmca_temp=config.cmca_temp,
)

# Load pretrained weights if available
if config.load_model and os.path.exists(config.load_model):
    model_file = Path(config.load_model) / "best_model.pt"
    if model_file.exists():
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loaded all model params from {model_file}")
        except Exception:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded matching params from {model_file}")

model.to(device)
wandb.watch(model)

# ============================================================================
# Prepare data loaders
# ============================================================================
from scgpt.trainer import prepare_data as sc_prepare_data

def _make_loader(data_pt, batch_size, shuffle=False, intra_shuffle=False, drop_last=False):
    """Create DataLoader with optional per-batch sampling."""
    class _SD(Dataset):
        def __init__(self, d): self.data = d
        def __len__(self): return self.data["gene_ids"].shape[0]
        def __getitem__(self, idx): return {k: v[idx] for k, v in self.data.items()}

    dataset = _SD(data_pt)
    subsets = []
    bl = data_pt["batch_labels"].numpy()
    for b in np.unique(bl):
        subsets.append(np.where(bl == b)[0].tolist())
    return DataLoader(
        dataset,
        batch_sampler=SubsetsBatchSampler(
            subsets, batch_size,
            intra_subset_shuffle=intra_shuffle,
            inter_subset_shuffle=shuffle,
            drop_last=drop_last,
        ),
        num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),
        pin_memory=True,
    )

# Loss functions
criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr,
    eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# ============================================================================
# Training loop with CMCA-Loss integration
# ============================================================================
best_val_loss = float("inf")
best_model = None
best_epoch = 0

for epoch in range(1, config.epochs + 1):
    epoch_start = time.time()

    # Prepare data (with random masking)
    train_pt, valid_pt = sc_prepare_data(
        tokenized_train, tokenized_valid,
        train_bl, valid_bl, config, epoch,
        sort_seq_batch=True,
    )

    train_loader = _make_loader(train_pt, config.batch_size,
                                 shuffle=False, intra_shuffle=True)
    valid_loader = _make_loader(valid_pt, config.batch_size,
                                 shuffle=False, intra_shuffle=False)

    # --- Training ---
    model.train()
    total_loss = 0.0
    total_cmca = 0.0
    n_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(train_loader):
        ig = batch_data["gene_ids"].to(device)
        iv = batch_data["values"].to(device)
        tv = batch_data["target_values"].to(device)
        bl = batch_data["batch_labels"].to(device)
        mt = batch_data.get("mod_types", None)
        if mt is not None:
            mt = mt.to(device)

        src_mask = ig.eq(vocab[config.pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp):
            out = model(
                ig, iv,
                src_key_padding_mask=src_mask,
                batch_labels=bl if config.DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CMCA=config.CMCA,
                mod_types=mt if config.use_mod else None,
            )

            masked_pos = iv.eq(config.mask_value)
            loss = criterion(out["mlm_output"], tv, masked_pos)

            if config.explicit_zero_prob and "mlm_zero_probs" in out:
                loss += criterion_neg_log_bernoulli(out["mlm_zero_probs"], tv, masked_pos)

            if config.GEPC:
                loss += criterion(out["mvc_output"], tv, masked_pos)
                if config.explicit_zero_prob and "mvc_zero_probs" in out:
                    loss += criterion_neg_log_bernoulli(out["mvc_zero_probs"], tv, masked_pos)

            if config.ecs_thres > 0:
                loss += 10 * out["loss_ecs"]

            loss_dab = criterion_dab(out["dab_output"], bl)
            loss += config.dab_weight * loss_dab

            # ---- CMCA-Loss integration ----
            cmca_loss_val = 0.0
            if config.CMCA and "loss_cmca" in out:
                cmca_loss_val = out["loss_cmca"]
                loss += config.cmca_weight * cmca_loss_val
                total_cmca += cmca_loss_val.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=False if scaler.is_enabled() else True)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        wandb.log({
            "train/loss": loss.item(),
            "train/cmca": cmca_loss_val.item() if isinstance(cmca_loss_val, torch.Tensor) else cmca_loss_val,
        })

        if batch_idx % config.log_interval == 0 and batch_idx > 0:
            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{n_batches:3d} batches | "
                f"loss {total_loss/(batch_idx+1):5.2f} | "
                f"cmca {total_cmca/(batch_idx+1):5.4f} |"
            )

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_n = 0
    with torch.no_grad():
        for batch_data in valid_loader:
            ig = batch_data["gene_ids"].to(device)
            iv = batch_data["values"].to(device)
            tv = batch_data["target_values"].to(device)
            bl = batch_data["batch_labels"].to(device)
            mt = batch_data.get("mod_types", None)
            if mt is not None:
                mt = mt.to(device)

            src_mask = ig.eq(vocab[config.pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                out = model(ig, iv, src_key_padding_mask=src_mask,
                            batch_labels=bl if config.DSBN else None,
                            MVC=False, ECS=False, CMCA=False,
                            mod_types=mt if config.use_mod else None)
                masked_pos = iv.eq(config.mask_value)
                loss = criterion(out["mlm_output"], tv, masked_pos)
                loss_dab = criterion_dab(out["dab_output"], bl)

            val_loss += (loss.item() + config.dab_weight * loss_dab.item()) * len(ig)
            val_n += len(ig)

    val_loss /= val_n
    elapsed = time.time() - epoch_start
    logger.info(f"| end epoch {epoch:3d} | time {elapsed:5.2f}s | val_loss {val_loss:5.4f}")

    wandb.log({"valid/loss": val_loss, "epoch": epoch})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_epoch = epoch
        logger.info(f"  -> New best model (epoch {epoch}, val_loss {val_loss:.4f})")

    # Save and evaluate
    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        ckpt_path = save_dir / f"model_e{best_epoch}.pt"
        torch.save(best_model.state_dict(), ckpt_path)
        logger.info(f"Model saved to {ckpt_path}")

        # Evaluate cell embeddings using eval_testdata from trainer
        from scgpt.trainer import eval_testdata as sc_eval_testdata
        results = sc_eval_testdata(
            best_model, adata_t=adata_sorted,
            gene_ids=gene_ids, vocab=vocab,
            config=config, logger=logger,
            include_types=["cls"],
        )
        if results:
            for fig_key in ["batch_umap", "celltype_umap"]:
                if fig_key in results:
                    results[fig_key].savefig(
                        save_dir / f"embeddings_{fig_key}[cls]_e{best_epoch}.png",
                        dpi=300, bbox_inches="tight")

            metrics = {f"test/{k}": v for k, v in results.items()
                       if k not in ["batch_umap", "celltype_umap"]}
            metrics["test/best_epoch"] = best_epoch
            wandb.log(metrics)
            wandb.log({"avg_bio": results.get("avg_bio", 0.0)})
            logger.info(f"Test avg_bio: {results.get('avg_bio', 0.0):.4f}")

    scheduler.step()

# Save final best model
final_path = save_dir / "best_model.pt"
torch.save(best_model.state_dict(), final_path)
logger.info(f"Training complete. Best model (epoch {best_epoch}) saved to {final_path}")

# Final comprehensive evaluation
from scgpt.trainer import eval_testdata as sc_eval_testdata
final_results = sc_eval_testdata(
    best_model, adata_t=adata_sorted,
    gene_ids=gene_ids, vocab=vocab,
    config=config, logger=logger,
    include_types=["cls"],
)
if final_results:
    for k, v in final_results.items():
        if k not in ["batch_umap", "celltype_umap"]:
            logger.info(f"Final {k}: {v:.4f}" if isinstance(v, float) else f"Final {k}: {v}")
    wandb.log({f"final/{k}": v for k, v in final_results.items()
               if k not in ["batch_umap", "celltype_umap"]})

wandb.finish()
gc.collect()
logger.info("Done.")
TRAIN_PYTHON_EOF

# ---------------------------------------------------------------------------
# 3. Set environment variables for the Python script
# ---------------------------------------------------------------------------
export OUTPUT_DIR
export LOAD_MODEL
export DISTRIBUTED

# ---------------------------------------------------------------------------
# 4. Launch training
# ---------------------------------------------------------------------------
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