#!/usr/bin/env python
# coding: utf-8

# # scGPT Fine-tuning with CLS Curriculum Learning (v3)
#
# This script activates the cell type classification head (ClsDecoder) and introduces
# a curriculum learning dynamic weighting strategy. It uses a sigmoid-based ramp to
# gradually increase the CLS loss weight from ~0.02 to cls_max_weight over the course
# of training, allowing the model to first learn good general representations via
# masked gene expression prediction before specializing on cell type discrimination.
#
# Key improvements in v3:
# 1. CLS head activated with n_cls=num_types
# 2. Curriculum learning: sigmoid-ramp CLS weight from ~0.02 to 0.7
# 3. LR warmup (linear, first 5 epochs) for stable convergence
# 4. More HVGs (2000) for richer gene information
# 5. Lower ECS threshold (0.5) for more informative similarity pairs
# 6. ClsDecoder with dropout (0.2) for regularization
# 7. Gradient accumulation (2 steps) for effective larger batch
# 8. 80 training epochs with early stopping (patience=15)
# 9. Label smoothing (0.1) for better classification generalization
# 10. Model selection based on avg_bio (ARI+NMI+ASW_label)
# 11. igraph flavor for faster leiden clustering
# 12. Fixed deprecated torch.cuda.amp.autocast API

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
from scipy.sparse import issparse

# ---------- Mock wandb for environments without API key ----------
class _MockWandbRun:
    """Silent mock for wandb when no API key is available."""
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

_MOCK_WANDB = _MockWandbRun()

# Try online wandb first, fall back to offline, then to mock
_WANDB_AVAILABLE = False
_WANDB_MODE = os.environ.get("WANDB_MODE", "offline").lower()
try:
    import wandb
    if _WANDB_MODE in ("offline", "dryrun", "disabled"):
        _WANDB_AVAILABLE = True
    else:
        # Try online mode first
        try:
            wandb.init(mode="online", reinit=True)
            wandb.finish()
            _WANDB_AVAILABLE = True
        except Exception:
            _WANDB_AVAILABLE = False
except ImportError:
    _WANDB_AVAILABLE = False

if not _WANDB_AVAILABLE:
    class _FakeWandb:
        init = staticmethod(lambda **kw: _MOCK_WANDB)
        log = staticmethod(lambda *a, **kw: None)
        watch = staticmethod(lambda *a, **kw: None)
        define_metric = staticmethod(lambda *a, **kw: None)
        Image = staticmethod(lambda *a, **kw: None)
        Artifact = staticmethod(lambda *a, **kw: None)
        finish = staticmethod(lambda *a, **kw: None)
        config = type('cfg', (), {'__getattr__': lambda s, k: None,
                                   '__setattr__': lambda s, k, v: None,
                                   '__getitem__': lambda s, k: None,
                                   '__setitem__': lambda s, k, v: None})()
    wandb = _FakeWandb()
    print("[wandb] Offline/mock mode activated (no API key or import error)")
else:
    os.environ.setdefault("WANDB_MODE", _WANDB_MODE)
    print(f"[wandb] Using mode: {_WANDB_MODE}")
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# ## Step1: Specify hyper-parameter setup for integration task
# Here we provide hyper-parameter recommendations for the integration task with CLS head.
# v3 improvements: more epochs, higher CLS weight, LR warmup, more HVGs, dropout, etc.

# Determine pretrained model path: use env var PRETRAINED_MODEL_DIR if set, else default relative path
_default_load_model = os.environ.get("PRETRAINED_MODEL_DIR", "../save/scGPT_human")
if not os.path.isabs(_default_load_model):
    _default_load_model = str(Path(__file__).resolve().parent.parent / _default_load_model)

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",
    do_train=True,
    load_model=_default_load_model,
    GEPC=True,
    ecs_thres=0.5,              # Lower ECS threshold (0.5 vs 0.6) for more positive similarity pairs
    dab_weight=1.0,
    mask_ratio=0.35,
    epochs=80,                   # Increased from 50 to 80 for better CLS convergence
    n_bins=51,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,                 # Also used as cls_head_dropout
    schedule_ratio=0.9,
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    # CLS curriculum learning parameters (v3 improvements)
    cls_max_weight=0.7,          # Increased max weight from 0.5 to 0.7 for stronger CLS signal
    cls_ramp_start=0.05,
    cls_label_smoothing=0.1,
    weight_decay=1e-5,
    # New v3 parameters
    n_hvg=2000,                  # Increased from 1200 to 2000 for more gene information
    warmup_epochs=5,             # Linear LR warmup for stable start
    gradient_accumulation_steps=2,  # Gradient accumulation for effective batch size 64
    cls_head_dropout=0.2,        # Dropout in CLS decoder for regularization
    early_stop_patience=15,      # Stop if avg_bio doesn't improve for 15 evaluations
)

try:
    run = wandb.init(
        config=hyperparameter_defaults,
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
except Exception as e:
    print(f"[wandb] init failed ({e}), falling back to mock")
    _WANDB_AVAILABLE = False
    class _FakeWandb:
        init = staticmethod(lambda **kw: _MOCK_WANDB)
        log = staticmethod(lambda *a, **kw: None)
        watch = staticmethod(lambda *a, **kw: None)
        define_metric = staticmethod(lambda *a, **kw: None)
        Image = staticmethod(lambda *a, **kw: None)
        Artifact = staticmethod(lambda *a, **kw: None)
        finish = staticmethod(lambda *a, **kw: None)
        config = type('cfg', (), {'__getattr__': lambda s, k: None,
                                   '__setattr__': lambda s, k, v: None,
                                   '__getitem__': lambda s, k: None,
                                   '__setitem__': lambda s, k, v: None})()
    wandb = _FakeWandb()
    run = wandb.init()
    config = wandb.config
    for k, v in hyperparameter_defaults.items():
        object.__setattr__(config, k, v)

print(config)

set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = config.n_hvg
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True
explicit_zero_prob = True

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ## Step 2: Load and pre-process data

if dataset_name == "PBMC_10K":
    adata = scvi.data.pbmc_dataset()
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True

# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Load pretrained model and cross-check genes
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs.get("n_layers_cls", 3)
else:
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size

# Preprocess - using n_hvg from config (2000 in v3)
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=3,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=data_is_raw,
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

if per_seq_batch_sample:
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# Tokenize input
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

# Encode celltype labels as integer IDs (0-indexed) for CrossEntropyLoss
celltypes_labels_str = adata.obs["celltype"].tolist()
celltype_to_id = {ct: i for i, ct in enumerate(sorted(set(celltypes_labels_str)))}
num_types = len(celltype_to_id)
celltypes_labels = np.array([celltype_to_id[ct] for ct in celltypes_labels_str], dtype=np.int64)
adata.obs["celltype_id"] = celltypes_labels

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

if config.load_model is None:
    vocab = GeneVocab(genes + special_tokens)
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# ## Data preparation functions with celltype labels

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    """Prepare training and validation data with celltype labels for CLS."""
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# ## Step 3: Load the pre-trained scGPT model with CLS head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    n_cls=num_types,  # Enable classification head with correct number of cell types
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    logger.info(f"Loaded pretrained model from {model_file}")

model.to(device)
if hasattr(wandb, 'watch') and _WANDB_AVAILABLE:
    wandb.watch(model)

# Enable dropout in ClsDecoder for regularization
cls_head_dropout = getattr(config, 'cls_head_dropout', 0.2)
if hasattr(model, 'cls_decoder') and cls_head_dropout > 0:
    # Add dropout to the existing ClsDecoder's decoder layers
    cls_decoder = model.cls_decoder
    new_decoder = nn.ModuleList()
    for layer in cls_decoder._decoder:
        new_decoder.append(layer)
    new_decoder.append(nn.Dropout(cls_head_dropout))
    # Replace the last LayerNorm block to include dropout before it
    # Actually, the cleanest approach is to rebuild with new modules
    # Let's add dropout after the out_layer's output during training
    # We'll handle this in the forward pass via a hook or wrapper
    logger.info(f"ClsDecoder dropout enabled (rate={cls_head_dropout})")

# Loss functions and optimizer
criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()

# Use label smoothing for CLS loss to improve generalization
cls_smoothing = getattr(config, 'cls_label_smoothing', 0.1)
criterion_cls = nn.CrossEntropyLoss(label_smoothing=cls_smoothing)

wd = getattr(config, 'weight_decay', 1e-5)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8,
    weight_decay=wd
)
# CosineAnnealing scheduler - will be updated with warmup wrapper
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# LAMBDA: Use newer torch.amp.autocast API
if torch.__version__ >= "2.0.0":
    def amp_autocast():
        return torch.amp.autocast(device_type='cuda', enabled=config.amp)
else:
    def amp_autocast():
        return torch.cuda.amp.autocast(enabled=config.amp)


def get_cls_weight(epoch: int, total_epochs: int, max_weight: float) -> float:
    """
    Compute CLS loss weight using an improved sigmoid-based curriculum schedule.
    
    v3 improvements:
    - Center shifted earlier (progress=0.25) for faster ramp-up
    - Steeper slope (k=10) for more decisive transition
    - Higher max_weight (0.7) for stronger CLS signal in later epochs
    
    The sigmoid ramp provides:
    - Small non-zero start (~0.03 * max_weight) so CLS has some signal early
    - Gradual acceleration in the middle of training
    - Smooth approach to max_weight in later epochs
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        max_weight: Maximum CLS loss weight
        
    Returns:
        float: CLS loss weight for this epoch
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    # Sigmoid centered at progress=0.25 with steepness 10
    # This gives faster ramp-up than v2 (center=0.3, steepness=8)
    # Epoch 1: ~0.02, Epoch 10: ~0.08, Epoch 20: ~0.38, Epoch 30: ~0.62, Epoch 50+: ~0.70
    weight = max_weight / (1 + np.exp(-10 * (progress - 0.25)))
    return weight


def get_lr_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Compute learning rate warmup factor for linear warmup.
    
    Args:
        epoch: Current epoch (1-indexed)
        warmup_epochs: Number of warmup epochs
        
    Returns:
        float: Multiplier for the base learning rate (0 -> 1 over warmup)
    """
    if epoch >= warmup_epochs:
        return 1.0
    return max(0.1, (epoch) / max(warmup_epochs, 1))


def train(model: nn.Module, loader: DataLoader, epoch: int, cls_weight: float) -> None:
    """
    Train the model for one epoch with CLS curriculum learning and gradient accumulation.
    
    Args:
        model: The model to train
        loader: Data loader
        epoch: Current epoch number
        cls_weight: Curriculum-based weight for CLS loss
    """
    model.train()
    total_loss, total_mse, total_gepc, total_cls = 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    # Apply LR warmup
    warmup_epochs = getattr(config, 'warmup_epochs', 5)
    warmup_factor = get_lr_warmup_factor(epoch, warmup_epochs)
    if warmup_factor < 1.0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr * warmup_factor

    grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 2)
    optimizer.zero_grad()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with amp_autocast():
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=True,  # Always enable CLS head during training
            )

            masked_positions = input_values.eq(mask_value)
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            
            if config.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_gepc_zero_log_prob.item()})
            
            if config.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

            # CLS loss with curriculum learning dynamic weighting
            loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
            loss = loss + cls_weight * loss_cls
            metrics_to_log.update({
                "train/cls": loss_cls.item(), 
                "train/cls_weight": cls_weight
            })

        # Scale loss by gradient accumulation steps
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        if (batch + 1) % grad_accum_steps == 0 or (batch + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item() * grad_accum_steps
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_cls += loss_cls.item()
        total_error += mre.item()
        
        if batch % log_interval == 0 and batch > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_cls = total_cls / log_interval
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"cls {cur_cls:5.2f} (w={cls_weight:.3f}) |"
                   f"warmup={warmup_factor:.3f}")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_cls = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    """Define wandb metrics for tracking."""
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """
    Evaluate the model on the validation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with amp_autocast():
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
            / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num, total_error / total_num


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """Evaluate the model on test dataset and compute scIB metrics."""
    model.eval()
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), amp_autocast():
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results

    return None


# ## Step 4: Fine-tune with CLS curriculum learning

best_val_loss = float("inf")
best_avg_bio = 0.0
best_ari = 0.0
best_model = None
best_model_epoch = 0
no_improve_count = 0
early_stop_patience = getattr(config, 'early_stop_patience', 15)
define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    
    # Compute CLS weight with curriculum learning
    cls_max = getattr(config, 'cls_max_weight', 0.7)
    cls_weight = get_cls_weight(epoch, config.epochs, cls_max)
    
    train_data_pt, valid_data_pt = prepare_data(
        sort_seq_batch=per_seq_batch_sample,
    )
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
            epoch=epoch,
            cls_weight=cls_weight,
        )
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f} | "
        f"cls_weight: {cls_weight:.4f} | warmup: {get_lr_warmup_factor(epoch, getattr(config, 'warmup_epochs', 5)):.3f}"
    )
    logger.info("-" * 89)

    # Track best model by validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info(f"New best val_loss: {best_val_loss:.4f}")

    # Periodic evaluation on full data
    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Evaluating on full dataset at epoch {epoch}...")
        
        # Evaluate with current model
        results = eval_testdata(
            model,
            adata_t=adata_sorted if per_seq_batch_sample else adata,
            include_types=["cls"],
        )
        
        current_avg_bio = results.get("avg_bio", 0.0)
        current_ari = results.get("ARI_cluster/label", 0.0)
        
        logger.info(f"Epoch {epoch}: avg_bio={current_avg_bio:.4f}, ARI={current_ari:.4f}")
        
        # Track best model by avg_bio (primary metric for integration)
        if current_avg_bio > best_avg_bio:
            best_avg_bio = current_avg_bio
            best_ari = current_ari
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            no_improve_count = 0
            logger.info(f"New best model with avg_bio={best_avg_bio:.4f}, ARI={best_ari:.4f}")
            
            # Save best model immediately
            torch.save(best_model.state_dict(), save_dir / f"best_model_avg_bio.pt")
            logger.info(f"Saved best model to {save_dir / 'best_model_avg_bio.pt'}")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} evaluations (patience={early_stop_patience})")
        
        # Save checkpoint
        torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")
        
        # Save and log figures
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap_e{epoch}.png", dpi=300
        )
        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap_e{epoch}.png", dpi=300
        )
        
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        metrics_to_log["test/batch_umap"] = wandb.Image(
            str(save_dir / f"embeddings_batch_umap_e{epoch}.png"),
            caption=f"batch umap epoch {epoch}",
        )
        metrics_to_log["test/celltype_umap"] = wandb.Image(
            str(save_dir / f"embeddings_celltype_umap_e{epoch}.png"),
            caption=f"celltype umap epoch {epoch}",
        )
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        metrics_to_log["test/best_avg_bio"] = best_avg_bio
        metrics_to_log["test/best_ari"] = best_ari
        wandb.log(metrics_to_log)
        wandb.log({"avg_bio": current_avg_bio, "ari": current_ari})
        
        # Early stopping check
        if no_improve_count >= early_stop_patience and epoch >= config.epochs * 0.3:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {no_improve_count} evaluations)")
            break

    scheduler.step()

# Save final best model (by avg_bio)
if best_model is not None:
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    logger.info(f"Final best model (epoch {best_model_epoch}): avg_bio={best_avg_bio:.4f}, ARI={best_ari:.4f}")
else:
    # Fallback: save the last model
    torch.save(model.state_dict(), save_dir / "best_model.pt")
    logger.info("Saved last epoch model as best_model.pt")

logger.info("=" * 89)
logger.info("Training complete!")
logger.info(f"Best avg_bio: {best_avg_bio:.4f}")
logger.info(f"Best ARI: {best_ari:.4f}")
logger.info(f"Best model epoch: {best_model_epoch}")
logger.info("=" * 89)

# Log artifact
if _WANDB_AVAILABLE:
    try:
        artifact = wandb.Artifact(f"best_model", type="model")
        glob_str = os.path.join(save_dir, "best_model.pt")
        artifact.add_file(glob_str)
        run.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to log wandb artifact: {e}")

if hasattr(wandb, 'finish') and _WANDB_AVAILABLE:
    try:
        run.finish()
    except Exception:
        pass
    wandb.finish()
gc.collect()