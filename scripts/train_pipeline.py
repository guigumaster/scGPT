"""
scGPT Integration Fine-tuning Pipeline
========================================
Adaptive Curriculum Learning & Elastic Cell Similarity Framework

This script implements a complete training, validation, and evaluation pipeline
for scGPT single-cell multiomics integration with 5 key improvements:

1. Cosine annealing curriculum for DAR adversarial weight & gradient reversal
2. Elastic Cell Similarity (ECS) regularization
3. Transformer encoder extended from 4 to 8 layers with pre-norm
4. Dynamic mask ratio curriculum (0.25 -> 0.55)
5. Warmup cosine learning rate schedule

Data source: norman_2019.h5ad (perturbation dataset)
  - gemgroup (1-8) used as batch labels for integration
  - gene_program (7 categories) used as cell type labels

Usage:
    python scripts/train_pipeline.py --epochs 30 --batch_size 64
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import numpy as np
import scanpy as sc
import wandb
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from scgpt.utils import set_seed, eval_scib_metrics
from scgpt.trainer import (
    get_curriculum_mask_ratio,
    get_cosine_dab_weight,
    get_cosine_grad_reverse_lambda,
    get_warmup_cosine_lr_scheduler,
)

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"


def parse_args():
    parser = argparse.ArgumentParser(
        description="scGPT Integration Fine-tuning with Curriculum Learning"
    )
    # Core settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="norman_2019",
                        help="Dataset name")
    parser.add_argument("--do_train", action="store_true", default=True,
                        help="Whether to train the model")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to pretrained model checkpoint (None = train from scratch)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save outputs (auto-generated if None)")

    # Architecture - [Improvement 3] 8-layer encoder
    parser.add_argument("--nlayers", type=int, default=8,
                        help="Number of transformer encoder layers")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--layer_size", type=int, default=128,
                        help="Embedding/hidden dimension size")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--pre_norm", action="store_true", default=True,
                        help="Use pre-layer normalization")

    # Training settings
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.4,
                        help="Base mask ratio for MLM")
    parser.add_argument("--n_bins", type=int, default=51,
                        help="Number of expression bins")
    parser.add_argument("--n_hvg", type=int, default=1200,
                        help="Number of highly variable genes")

    # Objective flags
    parser.add_argument("--GEPC", action="store_true", default=True,
                        help="Enable masked value prediction for cell embeddings")
    parser.add_argument("--ecs_thres", type=float, default=0.8,
                        help="ECS threshold (0 to disable)")
    parser.add_argument("--dab_weight", type=float, default=1.0,
                        help="DAR adversarial weight")

    # [Improvement 1] DAR Curriculum
    parser.add_argument("--dab_weight_curriculum", action="store_true", default=True,
                        help="Enable cosine annealing for DAB weight")
    parser.add_argument("--dab_weight_min", type=float, default=0.1,
                        help="Minimum DAB weight during curriculum")
    parser.add_argument("--dab_weight_max", type=float, default=1.0,
                        help="Maximum DAB weight during curriculum")
    parser.add_argument("--grad_reverse_curriculum", action="store_true", default=True,
                        help="Enable cosine annealing for gradient reversal")
    parser.add_argument("--grad_reverse_lambda_min", type=float, default=0.1,
                        help="Minimum gradient reversal lambda")
    parser.add_argument("--grad_reverse_lambda_max", type=float, default=1.0,
                        help="Maximum gradient reversal lambda")

    # [Improvement 4] Mask Ratio Curriculum
    parser.add_argument("--mask_ratio_curriculum", action="store_true", default=True,
                        help="Enable dynamic mask ratio curriculum")
    parser.add_argument("--mask_ratio_start", type=float, default=0.25,
                        help="Starting mask ratio")
    parser.add_argument("--mask_ratio_end", type=float, default=0.55,
                        help="Ending mask ratio")
    parser.add_argument("--mask_ratio_warmup_epochs", type=int, default=15,
                        help="Epochs to reach end mask ratio")

    # [Improvement 5] LR Schedule
    parser.add_argument("--lr_warmup_epochs", type=int, default=3,
                        help="Number of warmup epochs")
    parser.add_argument("--use_warmup_cosine_lr", action="store_true", default=True,
                        help="Use warmup cosine LR scheduler")
    parser.add_argument("--schedule_ratio", type=float, default=0.9,
                        help="StepLR gamma (fallback)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05,
                        help="Minimum LR as ratio of base LR")

    # Infrastructure
    parser.add_argument("--fast_transformer", action="store_true", default=True,
                        help="Use flash attention")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--save_eval_interval", type=int, default=5,
                        help="Interval for saving and evaluating")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Logging interval in batches")
    parser.add_argument("--wandb_project", type=str, default="scGPT",
                        help="Wandb project name")
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"],
                        help="Wandb mode")

    return parser.parse_args()


def create_wandb_config(args):
    """Convert argparse namespace to wandb-compatible config dict."""
    return {
        "seed": args.seed,
        "dataset_name": args.dataset_name,
        "do_train": args.do_train,
        "load_model": args.load_model,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "n_bins": args.n_bins,
        "GEPC": args.GEPC,
        "ecs_thres": args.ecs_thres,
        "dab_weight": args.dab_weight,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "layer_size": args.layer_size,
        "nlayers": args.nlayers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "schedule_ratio": args.schedule_ratio,
        "save_eval_interval": args.save_eval_interval,
        "log_interval": args.log_interval,
        "fast_transformer": args.fast_transformer,
        "pre_norm": args.pre_norm,
        "amp": args.amp,
        "dab_weight_curriculum": args.dab_weight_curriculum,
        "dab_weight_min": args.dab_weight_min,
        "dab_weight_max": args.dab_weight_max,
        "grad_reverse_curriculum": args.grad_reverse_curriculum,
        "grad_reverse_lambda_min": args.grad_reverse_lambda_min,
        "grad_reverse_lambda_max": args.grad_reverse_lambda_max,
        "mask_ratio_curriculum": args.mask_ratio_curriculum,
        "mask_ratio_start": args.mask_ratio_start,
        "mask_ratio_end": args.mask_ratio_end,
        "mask_ratio_warmup_epochs": args.mask_ratio_warmup_epochs,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "use_warmup_cosine_lr": args.use_warmup_cosine_lr,
        "min_lr_ratio": args.min_lr_ratio,
    }


###############################################################################
# Dataset & DataLoader
###############################################################################

class SeqDataset(Dataset):
    """PyTorch Dataset for scGPT tokenized data."""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    """Create DataLoader with optional per-sequence-batch sampling."""
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
                subsets, batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers, pin_memory=True,
    )


###############################################################################
# Training Functions
###############################################################################

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    config,
    vocab,
    pad_token,
    mask_value,
    device,
    criterion,
    criterion_dab,
    optimizer,
    scheduler,
    scaler,
    logger,
    epoch: int,
    total_epochs: int,
    DSBN: bool,
    explicit_zero_prob: bool,
) -> None:
    """
    Train model for one epoch with curriculum learning.
    
    [Improvement 1] Cosine annealing DAB weight & gradient reversal lambda
    [Improvement 2] ECS regularization
    """
    model.train()

    # Curriculum: DAB weight and gradient reversal coefficient
    current_dab_weight = get_cosine_dab_weight(config, epoch, total_epochs)
    current_grad_lambda = get_cosine_grad_reverse_lambda(config, epoch, total_epochs)

    if model.do_dab:
        model.grad_reverse_discriminator.set_lambda(current_grad_lambda)

    total_loss = 0.0
    total_mse = 0.0
    total_gepc = 0.0
    total_ecs = 0.0
    total_dab = 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
            )

            masked_positions = input_values.eq(mask_value)
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics = {"train/mse": loss_mse.item()}

            if explicit_zero_prob:
                loss_nzlp = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_nzlp
                metrics["train/nzlp"] = loss_nzlp.item()

            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics["train/mvc"] = loss_gepc.item()

            if config.GEPC and explicit_zero_prob:
                loss_mvc_nzlp = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_nzlp
                metrics["train/mvc_nzlp"] = loss_mvc_nzlp.item()

            # [Improvement 2] Elastic Cell Similarity
            if config.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics["train/ecs"] = loss_ecs.item()

            # [Improvement 1] DAR with curriculum weight
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + current_dab_weight * loss_dab
            metrics["train/dab"] = loss_dab.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
        scaler.step(optimizer)
        scaler.update()

        # Log curriculum params
        metrics["train/dab_weight"] = current_dab_weight
        metrics["train/grad_lambda"] = current_grad_lambda

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_ecs += loss_ecs.item() if config.ecs_thres > 0 else 0.0
        total_dab += loss_dab.item()
        total_error += mre.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else config.lr
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {total_loss / log_interval:5.2f} | "
                f"mse {total_mse / log_interval:5.2f} | "
                + (f"gepc {total_gepc / log_interval:5.2f} |" if config.GEPC else "")
                + (f"ecs {total_ecs / log_interval:5.2f} |" if config.ecs_thres > 0 else "")
                + f"dab {total_dab / log_interval:5.2f} "
                f"(dw={current_dab_weight:.3f},gl={current_grad_lambda:.3f})"
            )
            total_loss = 0.0
            total_mse = 0.0
            total_gepc = 0.0
            total_ecs = 0.0
            total_dab = 0.0
            total_error = 0.0
            start_time = time.time()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config,
    vocab,
    pad_token,
    mask_value,
    device,
    criterion,
    criterion_dab,
    logger,
    epoch: int,
    DSBN: bool,
) -> Tuple[float, float, float]:
    """Evaluate model on validation set."""
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
                    input_gene_ids, input_values,
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

    avg_loss = total_loss / total_num
    avg_error = total_error / total_num
    avg_dab = total_dab / total_num

    return avg_loss, avg_error, avg_dab


def evaluate_full(
    model: nn.Module,
    adata_t,
    gene_ids,
    input_layer_key: str,
    max_seq_len: int,
    vocab,
    pad_token,
    config,
    device,
    logger,
    DSBN: bool,
) -> Dict:
    """Full evaluation with scIB metrics (test phase)."""
    model.eval()
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )
    batch_ids = np.array(adata_t.obs["batch_id"].tolist())

    logger.info("Running full evaluation with scIB metrics...")
    tokenized_all = tokenize_and_pad_batch(
        all_counts, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=-2, append_cls=True, include_zero_gene=True,
    )
    all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        cell_embeddings = model.encode_batch(
            all_gene_ids, all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=config.batch_size,
            batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
            time_step=0, return_np=True,
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
        logger.error(f"scIB metrics failed: {e}")

    return results


def print_final_results(results: Dict, logger):
    """Print formatted final evaluation results."""
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)

    bio_metrics = {
        "NMI": results.get("NMI_cluster/label", "N/A"),
        "ARI": results.get("ARI_cluster/label", "N/A"),
        "ASW (label)": results.get("ASW_label", "N/A"),
    }
    batch_metrics = {
        "PCR_batch": results.get("PCR_batch", "N/A"),
        "ASW (batch)": results.get("ASW_label/batch", "N/A"),
        "graph_conn": results.get("graph_conn", "N/A"),
    }

    logger.info("Biological Conservation:")
    for k, v in bio_metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("Batch Effect Removal:")
    for k, v in batch_metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"avg_bio: {results.get('avg_bio', 'N/A')}")
    logger.info("=" * 60)


###############################################################################
# Main Pipeline
###############################################################################

def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup save directory
    if args.save_dir is None:
        save_dir = PROJECT_ROOT / f"save/pipeline_{args.dataset_name}-{time.strftime('%b%d-%H-%M')}"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Arguments: {args}")

    # Save config
    config_dict = create_wandb_config(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Initialize W&B
    run = wandb.init(
        config=config_dict,
        project=args.wandb_project,
        reinit=True,
        mode=args.wandb_mode,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config

    logger.info("=" * 60)
    logger.info("scGPT Adaptive Curriculum Learning Pipeline")
    logger.info("Improvements enabled:")
    logger.info(f"  1. Cosine Annealing DAR Curriculum: {args.dab_weight_curriculum}")
    logger.info(f"  2. Elastic Cell Similarity (ECS): {'enabled (thres=' + str(args.ecs_thres) + ')' if args.ecs_thres > 0 else 'disabled'}")
    logger.info(f"  3. Transformer Layers: {args.nlayers} (pre-norm: {args.pre_norm})")
    logger.info(f"  4. Mask Ratio Curriculum: {args.mask_ratio_curriculum} ({args.mask_ratio_start} -> {args.mask_ratio_end})")
    logger.info(f"  5. Warmup Cosine LR: {args.use_warmup_cosine_lr}")
    logger.info("=" * 60)

    ###########################################################################
    # Data Loading & Preprocessing
    ###########################################################################
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_value = -1
    pad_value = -2
    n_input_bins = args.n_bins
    per_seq_batch_sample = True
    DSBN = True
    explicit_zero_prob = True

    # Load dataset from available h5ad file
    logger.info(f"Loading dataset: {args.dataset_name}")
    data_path = PROJECT_ROOT / "data" / f"{args.dataset_name}.h5ad"
    if not data_path.exists():
        # Fallback: try data/norman_2019.h5ad
        data_path = PROJECT_ROOT / "data" / "norman_2019.h5ad"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {data_path}. "
                f"Available h5ad files: {list(PROJECT_ROOT.glob('data/*.h5ad'))}"
            )
    logger.info(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"Loaded adata with shape {adata.shape}")

    # Create batch and celltype columns from available metadata
    # norman_2019 has gemgroup (1-8) as batch and gene_program as celltype
    if "gemgroup" in adata.obs.columns and "gene_program" in adata.obs.columns:
        adata.obs["str_batch"] = adata.obs["gemgroup"].astype(int).astype(str)
        adata.obs["celltype"] = adata.obs["gene_program"].astype(str)
        ori_batch_col = "str_batch"
    elif "batch" in adata.obs.columns:
        ori_batch_col = "batch"
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        if "celltype" not in adata.obs.columns and "cell_type" in adata.obs.columns:
            adata.obs["celltype"] = adata.obs["cell_type"].astype(str)
        elif "celltype" not in adata.obs.columns:
            adata.obs["celltype"] = "unknown"
    else:
        # Create synthetic batches for demonstration
        logger.warning("No batch column found, creating synthetic batches")
        n_batches = min(8, adata.n_obs // 100)
        adata.obs["str_batch"] = np.random.randint(0, n_batches, adata.n_obs).astype(str)
        adata.obs["celltype"] = "unknown"
        ori_batch_col = "str_batch"

    # Batch encoding
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    if "gene_name" not in adata.var.columns and "gene_symbols" in adata.var.columns:
        adata.var["gene_name"] = adata.var["gene_symbols"].tolist()
    elif "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.tolist()

    # Count layers
    data_is_raw = True
    if "counts" in adata.layers:
        input_layer_key = "counts"
        # Use raw counts
        adata.X = adata.layers["counts"].copy()
    else:
        input_layer_key = "X"

    # Build vocabulary from scratch (since no pretrained model)
    embsize = args.layer_size
    nhead_model = args.nhead
    nlayers_model = args.nlayers
    d_hid = args.layer_size
    num_batch_types = len(set(batch_id_labels))

    # Create gene vocabulary
    all_genes = adata.var["gene_name"].tolist()
    vocab = GeneVocab(all_genes + special_tokens)
    logger.info(f"Created vocabulary with {len(vocab)} tokens from {len(all_genes)} genes")
    vocab.set_default_index(vocab["<pad>"])

    # Get gene IDs
    gene_ids_in_vocab = np.array([
        1 if gene in vocab else -1 for gene in all_genes
    ])
    adata = adata[:, gene_ids_in_vocab >= 0]

    logger.info(f"After vocabulary filter: {adata.shape}")

    # Preprocessing (includes HVG selection which reduces gene count)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=args.n_hvg,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")

    # Compute gene_ids AFTER preprocessing (HVG selection may have reduced genes)
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    logger.info(f"After HVG selection: {adata.shape}, gene_ids length: {len(gene_ids)}")

    if per_seq_batch_sample:
        adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()
    else:
        adata_sorted = adata

    # Tokenize
    input_layer_key = "X_binned"
    all_counts = (
        adata_sorted.layers[input_layer_key].toarray()
        if issparse(adata_sorted.layers[input_layer_key])
        else adata_sorted.layers[input_layer_key]
    )
    celltypes_labels = np.array(adata_sorted.obs["celltype"].tolist())
    batch_ids = np.array(adata_sorted.obs["batch_id"].tolist())

    # Train/val split
    (train_data, valid_data, _, _, train_batch_labels, valid_batch_labels) = \
        train_test_split(all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True)

    max_seq_len = args.n_hvg + 1
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True,
    )

    logger.info(f"Train samples: {tokenized_train['genes'].shape[0]}, "
                f"Valid samples: {tokenized_valid['genes'].shape[0]}")

    ###########################################################################
    # Model Creation
    ###########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens, embsize, nhead_model, d_hid, nlayers_model,
        vocab=vocab, dropout=args.dropout,
        pad_token=pad_token, pad_value=pad_value,
        do_mvc=args.GEPC, do_dab=True,
        use_batch_labels=True, num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN, n_input_bins=n_input_bins,
        ecs_threshold=args.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=args.fast_transformer,
        pre_norm=args.pre_norm,
    )

    model.to(device)
    wandb.watch(model)

    ###########################################################################
    # Optimizer, Scheduler, Scaler
    ###########################################################################
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8
    )

    num_batches_per_epoch = max(1, len(tokenized_train['genes']) // args.batch_size)
    if args.use_warmup_cosine_lr:
        scheduler = get_warmup_cosine_lr_scheduler(
            optimizer,
            num_warmup_epochs=args.lr_warmup_epochs,
            num_training_epochs=args.epochs,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        logger.info(f"Using warmup cosine LR (warmup={args.lr_warmup_epochs} epochs)")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=args.schedule_ratio
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ###########################################################################
    # Training Loop
    ###########################################################################
    best_val_loss = float("inf")
    best_model = None
    best_model_epoch = 0

    # Wandb metric definitions
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # [Improvement 4] Curriculum-based masking
        current_mask_ratio = get_curriculum_mask_ratio(config, epoch, args.epochs)
        
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=current_mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=current_mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        logger.info(
            f"epoch {epoch:3d}, mask_ratio={current_mask_ratio:.3f}, "
            f"masked_train%={(masked_values_train == mask_value).sum() / max(1, (masked_values_train != pad_value).sum()):.4f}"
        )

        # Prepare data dicts
        train_data_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values_train,
            "target_values": tokenized_train["values"],
            "batch_labels": torch.from_numpy(train_batch_labels).long(),
        }
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": masked_values_valid,
            "target_values": tokenized_valid["values"],
            "batch_labels": torch.from_numpy(valid_batch_labels).long(),
        }

        # Sort by batch for per_seq_batch sampling
        if per_seq_batch_sample:
            train_sort = np.argsort(train_batch_labels)
            for k in train_data_pt:
                train_data_pt[k] = train_data_pt[k][train_sort]
            valid_sort = np.argsort(valid_batch_labels)
            for k in valid_data_pt:
                valid_data_pt[k] = valid_data_pt[k][valid_sort]

        train_loader = prepare_dataloader(
            train_data_pt, batch_size=args.batch_size,
            shuffle=False, intra_domain_shuffle=True, drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt, batch_size=args.batch_size,
            shuffle=False, intra_domain_shuffle=False, drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
        )

        # Training
        if args.do_train:
            train_epoch(
                model, train_loader, config, vocab, pad_token, mask_value,
                device, criterion, criterion_dab, optimizer, scheduler, scaler,
                logger, epoch, args.epochs, DSBN, explicit_zero_prob,
            )

        # Validation
        val_loss, val_mre, val_dab = evaluate(
            model, valid_loader, config, vocab, pad_token, mask_value,
            device, criterion, criterion_dab, logger, epoch, DSBN,
        )

        elapsed = time.time() - epoch_start
        logger.info("-" * 89)
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"val loss {val_loss:5.4f} | mre {val_mre:5.4f} | dab {val_dab:5.4f}")
        logger.info("-" * 89)

        # Log validation metrics
        wandb.log({
            "valid/mse": val_loss,
            "valid/mre": val_mre,
            "valid/dab": val_dab,
            "valid/sum_mse_dab": val_loss + config.dab_weight * val_dab,
            "epoch": epoch,
        })

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"New best model (epoch {epoch}, val_loss={val_loss:.4f})")

        # Periodic evaluation
        if epoch % args.save_eval_interval == 0 or epoch == args.epochs:
            save_path = save_dir / f"model_e{best_model_epoch}.pt"
            torch.save(best_model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

            # Full evaluation
            results = evaluate_full(
                best_model, adata_sorted if per_seq_batch_sample else adata,
                gene_ids, input_layer_key, max_seq_len, vocab, pad_token,
                config, device, logger, DSBN,
            )
            if results:
                print_final_results(results, logger)
                wandb.log({
                    "test/NMI": results.get("NMI_cluster/label", 0),
                    "test/ARI": results.get("ARI_cluster/label", 0),
                    "test/PCR_batch": results.get("PCR_batch", 0),
                    "test/avg_bio": results.get("avg_bio", 0),
                    "test/ASW_label": results.get("ASW_label", 0),
                    "test/ASW_batch": results.get("ASW_label/batch", 0),
                    "test/graph_conn": results.get("graph_conn", 0),
                    "epoch": epoch,
                })

        scheduler.step()

    ###########################################################################
    # Final Save & Report
    ###########################################################################
    final_path = save_dir / "best_model.pt"
    if best_model is not None:
        torch.save(best_model.state_dict(), final_path)
        logger.info(f"Final best model saved to {final_path}")
    else:
        logger.warning("No best model saved (model may not have trained)")

    # Final evaluation
    if best_model is not None:
        logger.info("\nRunning final comprehensive evaluation...")
        final_results = evaluate_full(
            best_model, adata_sorted if per_seq_batch_sample else adata,
            gene_ids, input_layer_key, max_seq_len, vocab, pad_token,
            config, device, logger, DSBN,
        )
        if final_results:
            print_final_results(final_results, logger)

            # Write results summary
            summary = {
                "best_epoch": best_model_epoch,
                "best_val_loss": best_val_loss,
                "PCR_batch": final_results.get("PCR_batch", None),
                "ARI": final_results.get("ARI_cluster/label", None),
                "NMI": final_results.get("NMI_cluster/label", None),
                "ASW_label": final_results.get("ASW_label", None),
                "ASW_batch": final_results.get("ASW_label/batch", None),
                "graph_conn": final_results.get("graph_conn", None),
                "avg_bio": final_results.get("avg_bio", None),
            }
            with open(save_dir / "results_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Results summary saved to {save_dir / 'results_summary.json'}")

    wandb.log({"test/best_model_epoch": best_model_epoch})
    run.finish()
    wandb.finish()
    gc.collect()
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()