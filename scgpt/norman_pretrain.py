"""
Norman CRISPR Perturbation Continual Pre-training
==================================================
Implements continual pre-training on the Norman perturbation dataset with
adaptive mask-ratio curriculum learning. This improves the model's understanding
of gene-gene interactions and perturbation dynamics, leading to better cell
embeddings (higher ARI / NMI) during fine-tuning on integration tasks.
"""

# =============================================================================
# Compatibility fix for Python 3.13 + old setuptools
# =============================================================================
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    import importlib.machinery
    pkgutil.ImpImporter = importlib.machinery.PathFinder

import copy
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scgpt import logger
from scgpt.data_sampler import SubsetsBatchSampler
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASK_VALUE = -1
PAD_VALUE = -2
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]

# Adaptive mask-ratio curriculum boundaries (can be overridden by config)
# When norman_mask_ratio_fixed > 0, a fixed mask ratio is used instead
MASK_RATIO_HIGH = 0.55   # early epochs: explore broad gene-gene dependencies
MASK_RATIO_LOW = 0.25    # later epochs: refine fine-grained patterns
CURRICULUM_SWITCH_EPOCH = 5  # switch after this many epochs


def get_adaptive_mask_ratio(epoch: int, total_epochs: int = 12,
                            config=None) -> float:
    """
    Adaptive mask-ratio curriculum.

    If config has norman_mask_ratio_fixed > 0, that fixed ratio is used.
    Otherwise, uses the standard curriculum:
      - epoch 1..CURRICULUM_SWITCH_EPOCH: high mask ratio = MASK_RATIO_HIGH
      - epoch CURRICULUM_SWITCH_EPOCH+1..total: linear decay to MASK_RATIO_LOW
    """
    # Use fixed mask ratio if provided via config
    if config is not None and hasattr(config, 'norman_mask_ratio_fixed'):
        if config.norman_mask_ratio_fixed > 0:
            return config.norman_mask_ratio_fixed
    if epoch <= CURRICULUM_SWITCH_EPOCH:
        return MASK_RATIO_HIGH
    ratio = MASK_RATIO_HIGH - (MASK_RATIO_HIGH - MASK_RATIO_LOW) * (
        (epoch - CURRICULUM_SWITCH_EPOCH) / (total_epochs - CURRICULUM_SWITCH_EPOCH)
    )
    return max(ratio, MASK_RATIO_LOW)


def prepare_norman_data(
    tokenized_train: Dict,
    tokenized_valid: Dict,
    train_batch_labels: np.ndarray,
    valid_batch_labels: np.ndarray,
    mask_ratio: float,
    epoch: int,
    sort_seq_batch: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Tokenize and mask training / validation splits for continual pre-training."""
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=MASK_VALUE,
        pad_value=PAD_VALUE,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=MASK_VALUE,
        pad_value=PAD_VALUE,
    )
    logger.info(
        f"  [Norman continual pre-train epoch {epoch:3d}] "
        f"mask ratio = {mask_ratio:.2f}, "
        f"train masked fraction = "
        f"{(masked_values_train == MASK_VALUE).sum() / (masked_values_train - PAD_VALUE).count_nonzero():.4f}"
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

    train_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }
    return train_pt, valid_pt


class SeqDataset(Dataset):
    """Minimal dataset wrapper for dictionary-based data."""

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
    """Build a DataLoader, optionally with per-batch sampling."""
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        return DataLoader(
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
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )


def continual_pretrain_norman(
    model: nn.Module,
    vocab: GeneVocab,
    gene_ids: np.ndarray,
    adata: AnnData,
    config,
    device: torch.device,
    save_dir: Path,
    norman_epochs: int = 12,
    per_seq_batch_sample: bool = True,
    DSBN: bool = True,
    num_batch_types: int = 1,
) -> nn.Module:
    """
    Continual pre-training on the Norman CRISPR perturbation dataset with
    adaptive mask-ratio curriculum learning.

    The Norman dataset contains single/double gene knockout scRNA-seq data
    in K562 cells. Training on this data helps the model learn:
      - Gene-gene perturbation interactions
      - Fine-grained expression pattern changes
      - Robust gene representations transferable to integration tasks

    Parameters
    ----------
    model : nn.Module
        The scGPT Transformer model (already loaded with pretrained weights).
    vocab : GeneVocab
        Vocabulary mapping gene names to indices.
    gene_ids : np.ndarray
        Array of gene IDs matching the vocabulary.
    adata : AnnData
        Preprocessed Norman AnnData object (with ``X_binned`` layer).
    config : wandb.config or dict-like
        Hyper-parameter container.
    device : torch.device
    save_dir : Path
        Directory for saving intermediate checkpoints.
    norman_epochs : int
        Total number of continual pre-training epochs (default 12).
    per_seq_batch_sample : bool
    DSBN : bool
    num_batch_types : int

    Returns
    -------
    nn.Module
        The model after continual pre-training (best validation checkpoint).
    """
    logger.info("=" * 60)
    logger.info("Starting Norman continual pre-training with adaptive mask ratio")
    logger.info(f"  Epochs: {norman_epochs}")
    logger.info(f"  Mask ratio schedule: high={MASK_RATIO_HIGH} (epochs 1-{CURRICULUM_SWITCH_EPOCH}), "
                f"linear decay to {MASK_RATIO_LOW} (epochs {CURRICULUM_SWITCH_EPOCH+1}-{norman_epochs})")
    logger.info("=" * 60)

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # train / valid split
    train_data, valid_data, train_batch_labels, valid_batch_labels = train_test_split(
        all_counts, batch_ids, test_size=0.1, shuffle=True, random_state=config.seed
    )

    # Cap the maximum sequence length
    _max_seq_len = min(adata.n_vars + 1, 1201)
    logger.info(
        f"Norman max_seq_len set to {_max_seq_len} "
        f"(adata has {adata.n_vars} genes)"
    )
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=_max_seq_len, vocab=vocab,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
        append_cls=True, include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=_max_seq_len, vocab=vocab,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
        append_cls=True, include_zero_gene=True,
    )

    criterion = masked_mse_loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr * 0.5, eps=1e-4 if config.amp else 1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=norman_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, norman_epochs + 1):
        mask_ratio = get_adaptive_mask_ratio(epoch, norman_epochs, config=config)
        epoch_start = time.time()

        train_pt, valid_pt = prepare_norman_data(
            tokenized_train, tokenized_valid,
            train_batch_labels, valid_batch_labels,
            mask_ratio=mask_ratio, epoch=epoch,
            sort_seq_batch=per_seq_batch_sample,
        )
        train_loader = prepare_dataloader(
            train_pt, batch_size=config.batch_size,
            shuffle=False, intra_domain_shuffle=True, drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
        )
        valid_loader = prepare_dataloader(
            valid_pt, batch_size=config.batch_size,
            shuffle=False, intra_domain_shuffle=False, drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
        )

        # --- training ---
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = config.log_interval if hasattr(config, 'log_interval') else 100
        start_time = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])
            # The model was initialized with use_batch_labels=True, so we MUST
            # always pass batch_labels (even when DSBN is False).
            with torch.amp.autocast(device_type='cuda', enabled=config.amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,  # always pass; model expects them
                    MVC=getattr(config, "GEPC", True),
                    ECS=getattr(config, "ecs_thres", 0) > 0,
                )

                masked_positions = input_values.eq(MASK_VALUE)
                loss = loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                if getattr(config, "explicit_zero_prob", True):
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                if getattr(config, "GEPC", True):
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    loss_gepc_item = loss_gepc.item()
                else:
                    loss_gepc_item = 0.0

                # Add ECS loss for better cell embeddings during pre-training
                if getattr(config, "ecs_thres", 0) > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc_item
            total_error += mre.item()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                lr = optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                logger.info(
                    f"[Norman] epoch {epoch:3d}/{norman_epochs} | "
                    f"batch {batch_idx:3d}/{len(train_loader):3d} | "
                    f"lr {lr:.6f} | {ms_per_batch:5.2f} ms/batch | "
                    f"loss {total_loss/log_interval:5.2f} | "
                    f"mse {total_mse/log_interval:5.2f} | "
                    f"mre {total_error/log_interval:5.2f}"
                    + (f" | gepc {total_gepc/log_interval:5.2f}" if getattr(config, "GEPC", True) else "")
                )
                total_loss, total_mse, total_gepc, total_error = 0.0, 0.0, 0.0, 0.0
                start_time = time.time()

        # --- validation ---
        model.eval()
        val_loss, val_num = 0.0, 0
        with torch.no_grad():
            for batch_data in valid_loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])
                with torch.amp.autocast(device_type='cuda', enabled=config.amp):
                    output_dict = model(
                        input_gene_ids, input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels,  # always pass
                    )
                    masked_positions = input_values.eq(MASK_VALUE)
                    loss = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                val_loss += loss.item() * len(input_gene_ids)
                val_num += len(input_gene_ids)

        val_loss /= val_num
        elapsed = time.time() - epoch_start
        logger.info(
            f"[Norman] --- end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:.4f} | mask_ratio {mask_ratio:.2f} ---"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"[Norman] New best model with val_loss {best_val_loss:.4f}")

        scheduler.step()

    # restore best model
    model.load_state_dict(best_model_state)
    # save checkpoint
    norman_ckpt = save_dir / "norman_continual_pretrain.pt"
    torch.save(best_model_state, norman_ckpt)
    logger.info(f"[Norman] Continual pre-training done. Best model saved to {norman_ckpt}")

    return model