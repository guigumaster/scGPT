#!/usr/bin/env python
# coding: utf-8

"""
Norman Perturb-seq Data Continual Pretraining for scGPT
========================================================

This script performs continual pretraining of the scGPT model on the Norman 
Perturb-seq dataset (105 CRISPR gene perturbations). The pretraining uses only 
MLM (Masked Language Model) and GEPC/MVC (Gene Expression Prediction for Cell 
embedding) objectives, without batch-correction-specific objectives.

The key idea is that the diverse transcriptomic states induced by 105 different
gene perturbations expose the model to a much wider range of gene co-expression 
patterns than normal cells, leading to:
  1. Finer-grained understanding of gene regulatory relationships
  2. More discriminative cell embeddings that better separate cell types/states
  3. Improved downstream fine-tuning performance on integration tasks (higher ARI)

Usage:
    python tutorials/scripts/norman_continual_pretrain.py \
        --load_model /path/to/scGPT_human \
        --save_dir ./save/norman_enhanced \
        --epochs 20 \
        --batch_size 32 \
        --lr 5e-5
"""

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

# Add project root to sys.path.
# Script location: tutorials/scripts/norman_continual_pretrain.py
# Project root: two levels up from the script directory.
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer.vocab_compat import BuiltinVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

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

# Module-level wandb_run handle (set by main())
wandb_run = None


# ============================================================================
# Hyperparameters for Norman Continual Pretraining
# ============================================================================
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="Norman_Perturb",
    # Path to original pretrained scGPT model (whole-human recommended)
    load_model="../save/scGPT_human",
    # Continual pretraining objectives (only MLM + GEPC/MVC as specified)
    GEPC=True,            # Masked value prediction for cell embedding (MVC)
    ecs_thres=0.0,        # Disable ECS during continual pretraining
    dab_weight=0.0,       # Disable DAB during continual pretraining
    mask_ratio=0.35,      # Slightly lower mask ratio for more stable training
    epochs=10,            # Continual pretraining epochs
    n_bins=51,            # Number of bins for value binning
    lr=3e-5,              # Lower learning rate for more stable continual pretraining
    min_lr=1e-6,          # Minimum learning rate for cosine annealing
    batch_size=32,        # Batch size
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.15,         # Slightly lower dropout
    schedule_ratio=0.9,
    save_eval_interval=3,
    log_interval=50,
    fast_transformer=True,
    pre_norm=False,
    amp=True,             # Automatic Mixed Precision
    weight_decay=1e-5,    # Add weight decay for generalization
    gradient_clip=1.0,    # Gradient clipping
    # Norman data specific
    n_hvg=1500,           # More HVG for richer representation
    n_top_perturbed_genes=200,  # Focus on top perturbed genes for better signal
)

# ============================================================================
# Data Loading: Norman Perturb-seq Dataset
# ============================================================================
def load_norman_perturb_data(
    data_path: Optional[str] = None,
    n_top_genes: int = 1200,
    n_bins: int = 51,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple:
    """
    Load and preprocess the Norman Perturb-seq dataset.
    
    The Norman dataset (Norman et al. 2019) contains scRNA-seq profiles of K562 
    cells under 105 different CRISPR-mediated gene perturbations. This diverse
    transcriptomic perturbation data helps the model learn richer gene 
    co-expression patterns.
    
    The function tries multiple data sources:
    1. scPerturb package (preferred)
    2. Local h5ad file
    3. Direct download from public repository
    
    Args:
        data_path: Optional path to local h5ad file
        n_top_genes: Number of highly variable genes to select
        n_bins: Number of bins for expression value binning
        batch_size: Batch size for DataLoader
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, valid_loader, vocab, gene_ids, model_configs)
    """
    import scanpy as sc
    import anndata
    
    logger = scg.logger
    
    # Try to load Norman Perturb-seq data
    adata = None
    
    # Strategy 1: Load from scPerturb if available
    try:
        import scperturb
        logger.info("Loading Norman data via scPerturb...")
        # The Norman dataset is available in scPerturb as "norman"
        adata = scperturb.load_dataset("norman")
        logger.info(f"Loaded Norman data from scPerturb: {adata.shape}")
    except (ImportError, Exception) as e:
        logger.info(f"scPerturb not available or failed: {e}")
    
    # Strategy 2: Load from local file
    if adata is None:
        candidate_paths = [
            data_path,
            "../data/norman/norman.h5ad",
            "../data/norman.h5ad",
            "./data/norman/norman.h5ad",
            "./data/norman.h5ad",
        ]
        for cp in candidate_paths:
            if cp and os.path.exists(cp):
                logger.info(f"Loading Norman data from {cp}...")
                adata = sc.read_h5ad(cp)
                break
    
    # Strategy 3: Download Norman data
    if adata is None:
        logger.info("Attempting to download Norman Perturb-seq data...")
        download_dir = Path("./data/norman")
        download_dir.mkdir(parents=True, exist_ok=True)
        data_file = download_dir / "norman_perturb.h5ad"
        
        if not data_file.exists():
            # Try to download from public repository
            # The Norman data is available from figshare
            url = "https://figshare.com/ndownloader/files/35609593"
            try:
                import urllib.request
                logger.info(f"Downloading Norman data from {url}...")
                urllib.request.urlretrieve(url, data_file)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Failed to download Norman data: {e}")
                # As a fallback, generate synthetic perturbation-like data
                # for testing purposes (this will be replaced with real data)
                logger.warning("Creating synthetic perturb data for testing. "
                               "Replace with real Norman data for actual training.")
                adata = _create_synthetic_perturb_data()
                return _prepare_dataloaders(adata, n_top_genes, n_bins, batch_size, seed)
        
        if data_file.exists():
            adata = sc.read_h5ad(str(data_file))
    
    if adata is None:
        raise FileNotFoundError(
            "Could not load Norman Perturb-seq data. Please download it manually "
            "from https://figshare.com/ndownloader/files/35609593 and place it at "
            "./data/norman/norman_perturb.h5ad"
        )
    
    return _prepare_dataloaders(adata, n_top_genes, n_bins, batch_size, seed)


def _prepare_dataloaders(adata, n_top_genes, n_bins, batch_size, seed):
    """Prepare dataloaders from AnnData object."""
    logger = scg.logger
    
    set_seed(seed)
    
    logger.info(f"Norman data shape: {adata.shape}")
    logger.info(f"Number of perturbations: {len(adata.obs['perturbation'].unique()) if 'perturbation' in adata.obs else 'unknown'}")
    
    # Ensure raw counts are available
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    
    data_is_raw = True
    # Check if data is already log-normalized
    if adata.X.max() < 30:
        # Already log-transformed, need raw counts
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"]
    
    # Add required columns
    if "celltype" not in adata.obs:
        # Use perturbation as pseudo-celltype for diversity
        if "perturbation" in adata.obs:
            adata.obs["celltype"] = adata.obs["perturbation"].astype("category")
        else:
            adata.obs["celltype"] = "norman_cell"
    
    # Set up batch info
    if "batch" not in adata.obs:
        adata.obs["batch"] = "norman"
    adata.obs["str_batch"] = adata.obs["batch"].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    
    # Preprocess (binning only, no HVG selection yet since we need vocab first)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=n_top_genes,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")
    
    genes = adata.var["gene_name"].tolist()
    input_layer_key = "X_binned"
    
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    
    celltypes_labels = np.array(adata.obs["celltype"].tolist())
    batch_ids = np.array(adata.obs["batch_id"].tolist())
    
    # Split into train/valid
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
    
    logger.info(
        f"Train set: {train_data.shape[0]}, Valid set: {valid_data.shape[0]}"
    )
    
    return (
        train_data, valid_data,
        train_celltype_labels, valid_celltype_labels,
        train_batch_labels, valid_batch_labels,
        genes, adata,
    )


def _create_synthetic_perturb_data():
    """Create synthetic perturbation-like data for testing."""
    import scanpy as sc
    import numpy as np
    import pandas as pd
    
    n_cells = 5000
    n_genes = 2000
    
    # Generate baseline expression
    np.random.seed(42)
    X = np.random.negative_binomial(2, 0.01, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add perturbation effects for 105 different perturbations
    n_perturbs = 105
    perturbs = [f"perturb_{i}" for i in range(n_perturbs)]
    perturb_labels = np.random.choice(perturbs, size=n_cells)
    
    # Add perturbation-specific expression patterns
    for i, pert in enumerate(perturbs):
        mask = perturb_labels == pert
        n_masked = mask.sum()
        if n_masked > 0:
            # Each perturbation affects a random set of target genes
            target_genes = np.random.choice(n_genes, size=50, replace=False)
            # Some genes go up, some go down
            effects = np.random.normal(loc=0, scale=3, size=len(target_genes))
            X[mask][:, target_genes] += effects[np.newaxis, :]
    
    X = np.clip(X, 0, None)
    
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame({
            "perturbation": perturb_labels,
            "celltype": perturb_labels,
            "batch": "norman",
        }),
        var=pd.DataFrame(index=[f"GENE_{i}" for i in range(n_genes)]),
    )
    adata.layers["counts"] = adata.X.copy()
    
    return adata


# ============================================================================
# Data Preparation for Continual Pretraining
# ============================================================================
def prepare_pretrain_data(
    tokenized_train,
    tokenized_valid,
    train_batch_labels,
    valid_batch_labels,
    config,
    epoch,
    sort_seq_batch=False,
) -> Tuple[Dict[str, torch.Tensor]]:
    """Prepare training/validation data with random masking."""
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=config.mask_ratio,
        mask_value=-1,
        pad_value=-2,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=config.mask_ratio,
        mask_value=-1,
        pad_value=-2,
    )
    logger = scg.logger
    if epoch <= 2 or epoch % 5 == 0:
        ratio = (masked_values_train == -1).sum() / (masked_values_train - (-2)).count_nonzero()
        logger.info(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: {ratio:.4f}"
        )

    input_gene_ids_train = tokenized_train["genes"]
    input_gene_ids_valid = tokenized_valid["genes"]
    input_values_train = masked_values_train
    input_values_valid = masked_values_valid
    target_values_train = tokenized_train["values"]
    target_values_valid = tokenized_valid["values"]

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


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# ============================================================================
# Training Functions for Continual Pretraining
# ============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    config,
    device,
    vocab,
    epoch: int,
    criterion,
) -> Dict:
    """Train the model for one epoch with MLM and GEPC/MVC objectives."""
    global wandb_run
    model.train()
    total_loss, total_mlm, total_gepc = 0.0, 0.0, 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)
    logger = scg.logger

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,  # No batch labels during continual pretraining
                MVC=config.GEPC,    # Enable GEPC/MVC objective
                ECS=False,          # Disable ECS
            )

            masked_positions = input_values.eq(-1)
            
            # MLM loss
            loss_mlm = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            loss = loss_mlm
            
            metrics = {"train/mlm": loss_mlm.item()}
            
            # Explicit zero probability loss
            if hasattr(model, 'explicit_zero_prob') and model.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics["train/nzlp"] = loss_zero_log_prob.item()
            
            # GEPC (MVC) objective
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics["train/mvc"] = loss_gepc.item()
                
                if hasattr(model, 'explicit_zero_prob') and model.explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics["train/mvc_nzlp"] = loss_gepc_zero_log_prob.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_value = getattr(config, 'gradient_clip', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        scaler.step(optimizer)
        scaler.update()

        if wandb_run is not None:
            try:
                wandb.log(metrics)
            except Exception:
                pass

        total_loss += loss.item()
        total_mlm += loss_mlm.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0

        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mlm = total_mlm / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mlm {cur_mlm:5.2f} | "
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mlm = 0
            total_gepc = 0
            start_time = time.time()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device,
    vocab,
    config,
) -> Tuple[float, float]:
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    total_num = 0
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    MVC=False,
                    ECS=False,
                )
                
                masked_positions = input_values.eq(-1)
                
                # Only MLM loss for evaluation
                loss = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    avg_loss = total_loss / total_num
    return avg_loss


# ============================================================================
# Main Continual Pretraining Pipeline
# ============================================================================
def main():
    config = hyperparameter_defaults
    if isinstance(config, dict):
        # Convert dict to object-like access
        class Config:
            def __init__(self, d):
                self.__dict__.update(d)
        config = Config(config)
    
    # Allow command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description="Norman Continual Pretraining")
    parser.add_argument("--load_model", type=str, default=config.load_model,
                       help="Path to pretrained scGPT model")
    parser.add_argument("--save_dir", type=str, default="./save/norman_enhanced",
                       help="Directory to save enhanced model")
    parser.add_argument("--epochs", type=int, default=config.epochs,
                       help="Number of continual pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=config.lr,
                       help="Learning rate")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to Norman data h5ad file")
    parser.add_argument("--seed", type=int, default=config.seed,
                       help="Random seed")
    args = parser.parse_args()
    
    # Update config with args
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    set_seed(config.seed)
    
    logger = scg.logger
    logger.info("=" * 60)
    logger.info("Norman Perturb-seq Continual Pretraining")
    logger.info("=" * 60)
    logger.info(f"Config: {config.__dict__}")
    
    # Initialize wandb (if available) with error resilience
    global wandb_run
    wandb_run = None
    if _WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                config=config.__dict__,
                project="scGPT-Norman",
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Wandb init failed (continuing without): {e}")
            wandb_run = None
    else:
        logger.info("Wandb not available. Continuing without wandb logging.")
    
    # Create save directory
    timestamp = time.strftime('%b%d-%H-%M')
    save_dir = Path(f"{config.save_dir}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {save_dir}")
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    
    # ========================================================================
    # Step 1: Load Original Pretrained Model (or train from scratch)
    # ========================================================================
    logger.info(f"Loading pretrained model from {config.load_model}")
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    
    if model_dir.exists():
        # Load vocabulary from pretrained model
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        
        # Load model config
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
        _pretrained_available = True
        logger.info(f"✓ Loaded pretrained model config from {model_dir}")
    else:
        logger.warning(f"Pretrained model not found at {model_dir}. "
                      "Training from scratch (random initialization). "
                      "For best results, download the whole-human model.")
        # Use default hyperparameters
        embsize = config.layer_size
        nhead = config.nhead
        d_hid = config.layer_size * 4  # Default FF dimension
        nlayers = config.nlayers
        n_layers_cls = 3
        _pretrained_available = False
    
    # ========================================================================
    # Step 2: Load and Preprocess Norman Data
    # ========================================================================
    logger.info("Loading Norman Perturb-seq data...")
    
    result = load_norman_perturb_data(
        data_path=config.data_path,
        n_top_genes=config.n_hvg,
        n_bins=config.n_bins,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    
    (train_data, valid_data,
     train_celltype_labels, valid_celltype_labels,
     train_batch_labels, valid_batch_labels,
     genes, adata) = result
    
    # Build vocabulary from data genes (if no pretrained model)
    if not _pretrained_available:
        logger.info("Building vocabulary from data genes...")
        # Use BuiltinVocab directly to avoid torchtext compatibility issues
        vocab = BuiltinVocab(genes + special_tokens)
        vocab.set_default_index(vocab["<pad>"])
        logger.info(f"Built vocabulary with {len(vocab)} tokens")
    else:
        # Cross-check genes with vocabulary from pretrained model
        gene_names = genes
        gene_ids_in_vocab = [1 if g in vocab else -1 for g in gene_names]
        gene_ids_in_vocab = np.array(gene_ids_in_vocab)
        logger.info(
            f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        
        # Filter to common genes only
        keep_idx = gene_ids_in_vocab >= 0
        train_data = train_data[:, keep_idx]
        valid_data = valid_data[:, keep_idx]
        genes = [g for g, keep in zip(genes, keep_idx) if keep]
        logger.info(f"After filtering: {len(genes)} genes in common")
    
    # Tokenize
    max_seq_len = len(genes) + 1  # +1 for cls token
    gene_ids = np.array(vocab(genes), dtype=int)
    
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
    )
    
    logger.info(
        f"Train set: {tokenized_train['genes'].shape[0]} samples, "
        f"feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"Valid set: {tokenized_valid['genes'].shape[0]} samples, "
        f"feature length: {tokenized_valid['genes'].shape[1]}"
    )
    
    # ========================================================================
    # Step 3: Initialize Model (Load Pretrained Weights)
    # ========================================================================
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
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=config.GEPC,
        do_dab=False,           # No DAB during continual pretraining
        use_batch_labels=False,  # No batch labels
        domain_spec_batchnorm=False,
        n_input_bins=config.n_bins,
        ecs_threshold=0.0,
        explicit_zero_prob=True,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    )
    
    # Load pretrained weights (if available)
    if _pretrained_available:
        load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
        logger.info("Loaded pretrained weights successfully")
    else:
        logger.info("Training from scratch (random initialization)")
    
    model.to(device)
    if wandb_run is not None:
        try:
            wandb.watch(model)
        except Exception:
            pass
    
    # ========================================================================
    # Step 4: Setup Training
    # ========================================================================
    criterion = masked_mse_loss
    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        eps=1e-4 if config.amp else 1e-8,
        weight_decay=getattr(config, 'weight_decay', 1e-5),
    )
    # Use cosine annealing for smoother convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=getattr(config, 'min_lr', 1e-6),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    
    # ========================================================================
    # Step 5: Continual Pretraining Loop
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Starting Continual Pretraining")
    logger.info("=" * 60)
    
    best_val_loss = float("inf")
    best_model = None
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        
        # Prepare data with masking
        train_data_pt, valid_data_pt = prepare_pretrain_data(
            tokenized_train,
            tokenized_valid,
            train_batch_labels,
            valid_batch_labels,
            config,
            epoch,
        )
        
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        # Train one epoch
        train_epoch(
            model, train_loader, optimizer, scaler,
            config, device, vocab, epoch, criterion,
        )
        
        # Evaluate
        val_loss = evaluate(
            model, valid_loader, criterion, device, vocab, config,
        )
        
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 60)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:.6f}"
        )
        logger.info("-" * 60)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            logger.info(f"Best model at epoch {epoch} with val loss {val_loss:.6f}")
        
        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            # Save checkpoint
            ckpt_path = save_dir / f"norman_enhanced_e{epoch}.pt"
            torch.save(best_model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
        
        scheduler.step()
    
    # Save final best model
    final_path = save_dir / "best_model.pt"
    torch.save(best_model.state_dict(), final_path)
    logger.info(f"Saved best model to {final_path}")
    
    # Also save the model args and vocab for downstream fine-tuning
    with open(save_dir / "args.json", "w") as f:
        json.dump({
            "embsize": embsize,
            "nheads": nhead,
            "d_hid": d_hid,
            "nlayers": nlayers,
            "n_layers_cls": n_layers_cls,
            "vocab_size": ntokens,
            "n_input_bins": config.n_bins,
        }, f, indent=2)
    
    # Save vocab (handle both torchtext Vocab and BuiltinVocab)
    _save_vocab = getattr(vocab, 'save_json', None)
    if _save_vocab:
        _save_vocab(save_dir / "vocab.json")
    else:
        # BuiltinVocab does not have save_json - save manually
        itos = list(vocab.get_itos()) if hasattr(vocab, 'get_itos') else []
        if not itos:
            itos = [vocab[idx] for idx in range(len(vocab))]
        json.dump(itos, open(save_dir / "vocab.json", "w"))
        logger.info(f"Saved vocab ({len(itos)} tokens) to {save_dir / 'vocab.json'}")
    logger.info(f"Saved model config and vocabulary to {save_dir}")
    
    # Log artifact (wandb only)
    if wandb_run is not None:
        try:
            artifact = wandb.Artifact("norman_enhanced_model", type="model")
            artifact.add_file(str(final_path))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception as e:
            logger.warning(f"Wandb artifact logging failed: {e}")
    gc.collect()
    
    logger.info("=" * 60)
    logger.info("Continual Pretraining Complete!")
    logger.info(f"Enhanced model saved to: {save_dir}")
    logger.info(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
    logger.info("=" * 60)
    
    return save_dir


if __name__ == "__main__":
    main()