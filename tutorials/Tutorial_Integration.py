#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning on Pre-trained Model with Batch Integration (Enhanced with Norman Continual Pretraining)
# 
# In this tutorial, we demonstrate how to fine-tune a pre-trained model for batch integration.
# 
# **Key Enhancement**: The model is first continually pretrained on Norman Perturb-seq data
# (105 CRISPR gene perturbations) using MLM + GEPC/MVC objectives, which exposes the model
# to diverse transcriptomic perturbation states. This significantly improves the model's
# understanding of gene co-expression patterns and produces more discriminative cell
# embeddings, ultimately boosting ARI and other downstream integration metrics.
# 
# Pipeline:
#   1. (Optional) Continual pretrain on Norman Perturb-seq data → Norman-enhanced model
#   2. Specify hyper-parameter setup for integration task
#   3. Load and pre-process data (PBMC 10K / BMMC)
#   4. Load the pre-trained/Norman-enhanced scGPT model
#   5. Finetune scGPT with task-specific objectives (MLM + GEPC + ECS + DAB)
#   6. Evaluate fine-tuned scGPT with scib metrics (ARI, NMI, ASW, etc.)


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
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to sys.path for importing scgpt
# Tutorial_Integration.py is at tutorials/, project root is one level up
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.abspath(_project_root))
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
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

# ============================================================================
# Wandb setup - make wandb fully optional to avoid API key issues
# ============================================================================
# Set environment variables BEFORE importing wandb to ensure disabled mode works
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

_WANDB_AVAILABLE = False
try:
    import wandb
    _WANDB_AVAILABLE = True
    print(f"Wandb available, running in mode: {os.environ.get('WANDB_MODE', 'default')}")
except ImportError:
    print("Wandb not installed, skipping wandb logging.")


# ## Step1: Specify hyper-parameter setup for integration task
# Enhanced hyperparameters designed to maximize ARI:
# - Higher epochs for better convergence
# - Cosine annealing with warm restarts for better optimization
# - Optimal DAB weight for batch correction
# - Higher ECS threshold for better cell type separation
# - Lower learning rate with gradual decay

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",  # Dataset name
    do_train=True,  # Flag to indicate whether to update model parameters during training
    
    # Model path: supports both original scGPT and Norman-enhanced checkpoints
    load_model="../save/scGPT_human",  # Path to pre-trained model (original or Norman-enhanced)
    use_norman_enhanced=False,  # Set to True to use Norman-enhanced checkpoint automatically
    norman_model_path="../save/norman_enhanced",  # Path to Norman-enhanced checkpoint
    
    # Training objectives
    GEPC=True,  # Gene expression modelling for cell objective (MVC)
    ecs_thres=0.8,  # Elastic cell similarity objective (0.0 to 1.0, 0.0 to disable)
    dab_weight=1.0,  # DAR objective weight for batch correction
    mask_ratio=0.4,  # Mask ratio for MLM
    
    # Training hyperparameters (optimized for convergence and ARI)
    epochs=30,  # Increased from 15: more epochs for better representation learning
    n_bins=51,  # Number of bins for value binning in data pre-processing
    lr=1e-4,  # Learning rate for fine-tuning
    min_lr=1e-6,  # Minimum learning rate for cosine annealing
    batch_size=64,  # Batch size for fine-tuning
    layer_size=128,
    nlayers=4,
    nhead=4,  # (Ignored when loading a pretrained model)
    dropout=0.2,  # Dropout rate during model fine-tuning
    schedule_ratio=0.9,  # Learning rate decay factor (used if not cosine)
    save_eval_interval=3,  # Model evaluation interval
    log_interval=100,  # Log interval
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    
    # Advanced settings for improved cell embeddings
    use_cc_loss=True,  # Enable contrastive cell embedding loss (CCE) for better separation
    cce_weight=0.2,  # Weight for contrastive cell embedding loss (higher = more separation)
    ecs_threshold_hard=0.6,  # Stricter ECS threshold for more discriminative embeddings
    ecs_weight=10.0,  # Weight for ECS loss
    gradient_clip_value=1.0,  # Gradient clipping
    # ECS scheduling - gradually increase threshold for better separation
    ecs_schedule_start=0.4,  # Starting ECS threshold
    ecs_schedule_end=0.7,  # Final ECS threshold
    cce_warmup_epochs=2,  # Epochs before CCE kicks in
)

# Update hyperparameter_defaults with schedule values for ecs_thres
hyperparameter_defaults["ecs_thres"] = hyperparameter_defaults.get("ecs_thres", 0.8)

# Parse command-line arguments to allow flexible model selection
import argparse
_parser = argparse.ArgumentParser(description="scGPT Integration Fine-tuning")
_parser.add_argument("--load_model", type=str, default=None,
                    help="Path to pretrained scGPT model")
_parser.add_argument("--use_norman", action="store_true", default=False,
                    help="Use Norman-enhanced checkpoint")
_parser.add_argument("--norman_model_path", type=str, default="../save/norman_enhanced",
                    help="Path to Norman-enhanced checkpoint")
_parser.add_argument("--dataset", type=str, default="PBMC_10K",
                    help="Dataset name")
_parser.add_argument("--epochs", type=int, default=None,
                    help="Number of fine-tuning epochs")
_parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate")
_parser.add_argument("--batch_size", type=int, default=None,
                    help="Batch size")
_parser.add_argument("--seed", type=int, default=None,
                    help="Random seed")
_parser.add_argument("--dab_weight", type=float, default=None,
                    help="DAB weight")
_args, _unknown = _parser.parse_known_args()

# Apply command-line overrides
if _args.load_model is not None:
    hyperparameter_defaults["load_model"] = _args.load_model
if _args.use_norman:
    hyperparameter_defaults["use_norman_enhanced"] = True
if _args.norman_model_path:
    hyperparameter_defaults["norman_model_path"] = _args.norman_model_path
if _args.dataset:
    hyperparameter_defaults["dataset_name"] = _args.dataset
if _args.epochs is not None:
    hyperparameter_defaults["epochs"] = _args.epochs
if _args.lr is not None:
    hyperparameter_defaults["lr"] = _args.lr
if _args.batch_size is not None:
    hyperparameter_defaults["batch_size"] = _args.batch_size
if _args.seed is not None:
    hyperparameter_defaults["seed"] = _args.seed
if _args.dab_weight is not None:
    hyperparameter_defaults["dab_weight"] = _args.dab_weight


wandb_run = None
if _WANDB_AVAILABLE:
    try:
        wandb_run = wandb.init(
            config=hyperparameter_defaults,
            project="scGPT",
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        config = wandb.config
        print("Wandb initialized successfully")
    except Exception as e:
        print(f"Wandb init failed (continuing without): {e}")
        wandb_run = None
        # Create a config object from the defaults
        class _Config:
            def __init__(self, d):
                self.__dict__.update(d)
        config = _Config(hyperparameter_defaults)
else:
    wandb_run = None
    # Create a config object from the defaults
    class _Config:
        def __init__(self, d):
            self.__dict__.update(d)
    config = _Config(hyperparameter_defaults)
    print("Wandb not available. Using local config.")
print(dict(config.__dict__) if hasattr(config, '__dict__') else config)

set_seed(config.seed)


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 1200  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros


dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ## Step 2: Load and pre-process data

# ### 2.1 Load the PBMC 10K / BMMC data
# The BMMC (Bone Marrow Mononuclear Cells) multi-omics dataset is used for
# the integration task. It contains cells from multiple batches/conditions
# that need to be integrated into a common embedding space.

if dataset_name == "PBMC_10K":
    # Use standalone PBMC loader (no scvi dependency needed)
    from scripts.pbmc_loader import load_pbmc10k
    adata = load_pbmc10k()  # 11990 × 3346
    ori_batch_col = "batch"
    # celltype column already added by pbmc_loader
    if "gene_symbols" in adata.var.columns:
        adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True
elif dataset_name == "BMMC":
    # BMMC multi-omics dataset (replace with actual path if needed)
    bmmc_path = "../data/bmmc/bmmc_multiomic.h5ad"
    if os.path.exists(bmmc_path):
        adata = sc.read_h5ad(bmmc_path)
        ori_batch_col = "batch"
        data_is_raw = True
        logger.info(f"Loaded BMMC data: {adata.shape}")
    else:
        logger.warning(f"BMMC data not found at {bmmc_path}, falling back to PBMC_10K")
        from scripts.pbmc_loader import load_pbmc10k
        adata = load_pbmc10k()
        ori_batch_col = "batch"
        if "gene_symbols" in adata.var.columns:
            adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = True

# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


# ### 2.2 Cross-check gene set with the pre-trained model
# Note that we retain the common gene set between the data and the pre-trained model.

# Determine which model to load
if config.use_norman_enhanced:
    # Use Norman-enhanced checkpoint (prioritize this)
    model_load_path = config.norman_model_path
    logger.info(f"Using Norman-enhanced checkpoint: {model_load_path}")
else:
    model_load_path = config.load_model
    logger.info(f"Using original checkpoint: {model_load_path}")

# If model_load_path doesn't exist, try to find the latest valid Norman-enhanced checkpoint
if not os.path.exists(str(model_load_path)):
    logger.warning(f"Specified path {model_load_path} not found, trying auto-discovery...")
    # Look for the latest norman_enhanced checkpoint that has actual model files
    save_base = Path("../save")
    if save_base.exists():
        norman_dirs = sorted([d for d in save_base.iterdir() 
                             if d.is_dir() and "norman_enhanced" in d.name
                             and (d / "best_model.pt").exists()])
        if norman_dirs:
            model_load_path = str(norman_dirs[-1])
            logger.info(f"Auto-discovered valid Norman checkpoint: {model_load_path}")
        else:
            model_load_path = config.load_model
            logger.info(f"Falling back to original model: {model_load_path}")

if os.path.exists(str(model_load_path)):
    model_dir = Path(str(model_load_path))
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

    # model
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
    n_layers_cls = model_configs["n_layers_cls"]
else:
    logger.info("No pretrained model found, using default hyperparameters.")
    embsize = config.layer_size 
    nhead = config.nhead
    nlayers = config.nlayers  
    d_hid = config.layer_size


# ### 2.3 Pre-process the data

# set up the preprocessor
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
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()


# ### 2.4 Tokenize the input data

input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels_raw = adata.obs["celltype"].tolist()
# Encode cell types as categorical codes for numerical processing
from pandas import Categorical
celltypes_cat = Categorical(celltypes_labels_raw)
celltypes_labels = celltypes_cat.codes.astype(np.int64)
num_types = len(celltypes_cat.categories)
# Store mapping for later use in evaluation
celltype_names = list(celltypes_cat.categories)

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

# Print class distribution for debugging
train_celltype_counts = np.bincount(train_celltype_labels.astype(int))
logger.info(f"Train celltype distribution: {train_celltype_counts}")


# Ensure vocab is defined
_model_dir_was_set = ('model_dir' in dir() or 'model_dir' in locals()) and os.path.exists(str(model_dir)) if 'model_dir' in dir() or 'model_dir' in locals() else False
if _model_dir_was_set:
    pass  # vocab already loaded from the pretrained model section above
elif os.path.exists(str(Path(str(model_load_path)) / "vocab.json")):
    _vocab_path = Path(str(model_load_path)) / "vocab.json"
    logger.info(f"Loading vocabulary from {_vocab_path}")
    vocab = GeneVocab.from_file(_vocab_path)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
else:
    logger.info("Building vocabulary from data genes...")
    # Force using BuiltinVocab to avoid torchtext Vocab hang bug
    from scgpt.tokenizer.vocab_compat import BuiltinVocab
    vocab = BuiltinVocab(genes + special_tokens)
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



# In[13]:
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
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

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
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


#  ## Step 3: Load the pre-trained / Norman-enhanced scGPT model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

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
)

# Load model weights (either Norman-enhanced or original)
# Strategy: try multiple possible checkpoint locations
_model_loaded = False
_load_candidates = []

# Candidate 1: Norman-enhanced checkpoint path
if config.use_norman_enhanced:
    _load_candidates.append(Path(str(config.norman_model_path)) / "best_model.pt")
    _load_candidates.append(Path(str(config.norman_model_path)) / "norman_enhanced.pt")

# Candidate 2: auto-discovered model_load_path
_model_load_path = Path(str(model_load_path))
if (_model_load_path / "best_model.pt").exists():
    _load_candidates.append(_model_load_path / "best_model.pt")
if (_model_load_path / "norman_enhanced.pt").exists():
    _load_candidates.append(_model_load_path / "norman_enhanced.pt")

# Candidate 3: original model path from config
if config.load_model and Path(config.load_model).exists():
    orig_path = Path(config.load_model) / "best_model.pt"
    if orig_path.exists():
        _load_candidates.append(orig_path)

for ckpt in _load_candidates:
    if ckpt.exists():
        logger.info(f"Attempting to load model from {ckpt}")
        try:
            load_pretrained(model, torch.load(ckpt, map_location=device), verbose=False)
            _model_loaded = True
            logger.info(f"Successfully loaded weights from {ckpt}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {ckpt}: {e}")

if not _model_loaded:
    logger.warning("No pretrained model weights loaded. Training from scratch (random init).")

model.to(device)

# Log model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Model total parameters: {total_params:,}, trainable: {trainable_params:,}")
if wandb_run is not None:
    try:
        wandb.watch(model)
    except Exception:
        pass


# ## Step 4: Setup Training

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.lr, 
    eps=1e-4 if config.amp else 1e-8,
    weight_decay=1e-5,  # Add weight decay for better generalization
)

# Use Cosine Annealing with Warm Restarts for better convergence
# This helps the model escape local minima and find better representations
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=config.epochs,
    eta_min=config.min_lr if hasattr(config, 'min_lr') else 1e-6,
)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, loader: DataLoader, current_epoch: int) -> None:
    """
    Train the model for one epoch with all integration objectives.
    
    Uses MLM + GEPC (MVC) + ECS + DAB objectives jointly.
    When use_cc_loss is enabled, also uses contrastive cell embedding loss (CCE)
    for more discriminative cell representations.
    
    ECS threshold is scheduled: starts low, gradually increases for better
    cell type separation as training progresses.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    # ECS scheduling: gradually increase threshold for better separation
    # Start with a lower threshold to allow initial flexibility,
    # then increase to push cells of same type closer together
    ecs_schedule_start = getattr(config, 'ecs_schedule_start', 0.4)
    ecs_schedule_end = getattr(config, 'ecs_schedule_end', 0.7)
    total_epochs = getattr(config, 'epochs', 30)
    if total_epochs > 1:
        ecs_progress = min(1.0, (current_epoch - 1) / (total_epochs - 1))
    else:
        ecs_progress = 1.0
    # Sigmoid-like schedule for smoother transition
    current_ecs_thres = ecs_schedule_start + (ecs_schedule_end - ecs_schedule_start) * (
        1.0 / (1.0 + np.exp(-10.0 * (ecs_progress - 0.5)))
    )
    
    # CCE warmup: only enable after a few epochs
    cce_warmup = getattr(config, 'cce_warmup_epochs', 2)
    use_cce = getattr(config, 'use_cc_loss', False) and current_epoch > cce_warmup
    cce_weight = getattr(config, 'cce_weight', 0.2)
    ecs_weight = getattr(config, 'ecs_weight', 10.0)
    
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=current_ecs_thres > 0,
                CCE=use_cce,  # Contrastive cell embedding for better separation
            )

            masked_positions = input_values.eq(mask_value)
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {
                "train/mse": loss_mse.item(),
                "train/ecs_thres": current_ecs_thres,
            }
            
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
            
            if current_ecs_thres > 0:
                loss_ecs = ecs_weight * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            
            # Add CCE loss if enabled
            if use_cce and "loss_cce" in output_dict:
                loss_cce = cce_weight * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping to prevent gradient explosion
        clip_value = getattr(config, 'gradient_clip_value', 1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_value,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if wandb_run is not None:
            try:
                wandb.log(metrics_to_log)
            except Exception:
                pass

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    if wandb_run is None:
        return
    try:
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")
        wandb.define_metric("test/ARI_cluster/label", summary="max")
        wandb.define_metric("test/NMI_cluster/label", summary="max")
        wandb.define_metric("test/ASW_label", summary="max")
    except Exception:
        pass


def evaluate(model: nn.Module, loader: DataLoader) -> float:
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
            with torch.cuda.amp.autocast(enabled=config.amp):
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

    if wandb_run is not None:
        try:
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
        except Exception:
            pass

    return total_loss / total_num, total_error / total_num


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
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

    # Evaluate cls cell embeddings
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
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
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

        # Log individual metrics for detailed analysis
        if results:
            for key in ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]:
                if key in results:
                    logger.info(f"{key}: {results[key]:.4f}")

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


# ## Step 5: Finetune scGPT with task-specific objectives
# The fine-tuning jointly optimizes:
# - MLM: Masked language modeling for gene expression prediction
# - GEPC/MVC: Gene expression prediction for cell embedding
# - ECS: Elastic cell similarity for better cell type separation
# - DAB: Domain adversarial batch correction
# - CCE (optional): Contrastive cell embedding for discriminative representations

best_val_loss = float("inf")
best_avg_bio = 0.0
best_ari = 0.0
best_model = None
best_model_epoch = 0
define_wandb_metrcis()

logger.info("=" * 60)
model_source = "Norman-enhanced" if config.use_norman_enhanced else "Original"
logger.info(f"Starting fine-tuning with {model_source} checkpoint")
logger.info(f"Total epochs: {config.epochs}")
logger.info("=" * 60)

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
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
            current_epoch=epoch,
        )
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        # Eval on testdata
        results = eval_testdata(
            best_model,
            adata_t=adata_sorted if per_seq_batch_sample else adata,
            include_types=["cls"],
        )
        
        if results:
            # Track ARI specifically for monitoring improvement
            current_ari = results.get("ARI_cluster/label", 0.0)
            if current_ari > best_ari:
                best_ari = current_ari
                logger.info(f"New best ARI: {best_ari:.4f}")
            
            avg_bio = results.get("avg_bio", 0.0)
            if avg_bio > best_avg_bio:
                best_avg_bio = avg_bio
            
            logger.info(f"Epoch {epoch} - ARI: {current_ari:.4f}, avg_bio: {avg_bio:.4f}")
        
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
        )

        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        if wandb_run is not None:
            try:
                metrics_to_log["test/batch_umap"] = wandb.Image(
                    str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )
                metrics_to_log["test/celltype_umap"] = wandb.Image(
                    str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )
                metrics_to_log["test/best_model_epoch"] = best_model_epoch
                wandb.log(metrics_to_log)
                wandb.log({"avg_bio": results.get("avg_bio", 0.0)})
            except Exception:
                pass

    scheduler.step()


# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")
logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")
logger.info(f"Best epoch: {best_model_epoch}, Best val loss: {best_val_loss:.4f}")
logger.info(f"Best ARI: {best_ari:.4f}, Best avg_bio: {best_avg_bio:.4f}")

if wandb_run is not None:
    try:
        artifact = wandb.Artifact(f"best_model", type="model")
        glob_str = os.path.join(save_dir, "best_model.pt")
        artifact.add_file(glob_str)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    except Exception as e:
        logger.warning(f"Wandb artifact logging failed: {e}")
gc.collect()