#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning on Pre-trained Model with Batch Integration
# In this tutorial, we demonstrate how to fine-tune a pre-trained model on a new dataset for the batch integration task. We use the PBMC 10K dataset as an example and fine-tune on the pre-trained whole-body model. 
# 
# We summarize the fine-tuning pipeline in the following steps, which can be used as a general recipe for finetuning on integration tasks and beyond: 
# 
#      1. Specify hyper-parameter setup for integration task
#      
#      2. Load and pre-process data
#      
#      3. Load the pre-trained scGPT model
#      
#      4. Finetune scGPT with task-specific objectives
#      
#      5. Evaluate fine-tuned scGPT
# 

# In[2]:

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
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # already set
from anndata import AnnData
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Set project root: this script is at PROJECT_ROOT/tutorials/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable wandb (no API key available) - use offline mode as fallback
try:
    import wandb
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    WANDB_AVAILABLE = True
except (ImportError, AttributeError):
    WANDB_AVAILABLE = False
    print("Note: wandb not available, running without it.")

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
from scgpt.data import load_norman_data, preprocess_norman_data, prepare_norman_continual_pretrain_data
from scgpt.data import load_pbmc_dataset
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# =============================================================================
# Module-level class and function definitions
# These must be at module level for DataLoader workers (spawn multiprocessing)
# =============================================================================

class SimpleConfig:
    """Fallback config object when wandb is not available."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __contains__(self, key):
        return hasattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items() if not k.startswith('_')})


class DummyWandb:
    """Dummy wandb module when wandb is unavailable."""
    @staticmethod
    def log(*args, **kwargs): pass
    @staticmethod
    def watch(*args, **kwargs): pass
    @staticmethod
    def define_metric(*args, **kwargs): pass
    @staticmethod
    def Image(*args, **kwargs): return None
    @staticmethod
    def Artifact(*args, **kwargs): 
        class DummyArtifact:
            def add_file(self, *args, **kwargs): pass
        return DummyArtifact()
    class Settings:
        @staticmethod
        def start_method(*args, **kwargs): pass


class SeqDataset(Dataset):
    """Dataset for scGPT training data."""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_data(sort_seq_batch=False, epoch=0, mask_ratio_schedule=0.35,
                 tokenized_train=None, tokenized_valid=None,
                 train_batch_labels=None, valid_batch_labels=None,
                 mask_value=-1, pad_value=-2, mask_ratio=0.35):
    """
    Prepare training and validation data with dynamic masking.
    """
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio_schedule,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio_schedule,
        mask_value=mask_value,
        pad_value=pad_value,
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


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = True,
    config=None,
) -> DataLoader:
    """Create DataLoader with optional per-batch sampling."""
    dataset = SeqDataset(data_pt)

    if num_workers == 0 and config is not None and hasattr(config, 'num_workers') and config.num_workers > 0:
        n_workers = min(config.num_workers, 4)
    else:
        n_workers = num_workers

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
            num_workers=n_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=n_workers,
        pin_memory=True,
    )
    return data_loader


def train(model: nn.Module, loader: DataLoader, config, device, vocab, optimizer,
          scheduler, scaler, criterion, criterion_dab, mask_value, pad_token, epoch,
          mask_ratio, logger, wandb) -> None:
    """Train the model for one epoch with gradient accumulation."""
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device, non_blocking=True)
        input_values = batch_data["values"].to(device, non_blocking=True)
        target_values = batch_data["target_values"].to(device, non_blocking=True)
        batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)

        DSBN = True
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.amp.autocast('cuda', enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
            )

            masked_positions = input_values.eq(mask_value)
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            if config.explicit_zero_prob if hasattr(config, 'explicit_zero_prob') else True:
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
            if config.GEPC and (config.explicit_zero_prob if hasattr(config, 'explicit_zero_prob') else True):
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if config.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == num_batches:
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        loss_item = loss.item() * accumulation_steps
        total_loss += loss_item
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
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis(wandb):
    """Define wandb metrics."""
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, config, device, vocab,
             criterion, criterion_dab, mask_value, pad_token, epoch, wandb) -> Tuple[float, float]:
    """Evaluate the model on the evaluation data."""
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    DSBN = True
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device, non_blocking=True)
            input_values = batch_data["values"].to(device, non_blocking=True)
            target_values = batch_data["target_values"].to(device, non_blocking=True)
            batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.amp.autocast('cuda', enabled=config.amp):
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
    config,
    device,
    vocab,
    input_layer_key,
    gene_ids,
    max_seq_len,
    pad_token,
    pad_value,
    DSBN,
    logger,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """Evaluate the model on test dataset of adata_t."""
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
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=config.amp):
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


# =============================================================================
# Main training function - all runtime code is here
# Under `if __name__ == "__main__":` guard to prevent DataLoader workers
# from re-executing I/O and training code when spawned.
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Hyper-parameter Setup
    # -------------------------------------------------------------------------
    hyperparameter_defaults = dict(
        seed=42,
        dataset_name="PBMC_10K",
        do_train=True,
        load_model=str(PROJECT_ROOT / "save" / "scGPT_human"),
        GEPC=True,
        ecs_thres=0.5,
        dab_weight=0.3,
        mask_ratio=0.35,
        epochs=45,
        n_bins=51,
        lr=3e-5,
        batch_size=128,
        layer_size=128,
        nlayers=4,
        nhead=4,
        dropout=0.2,
        schedule_ratio=0.9,
        save_eval_interval=5,
        log_interval=200,
        fast_transformer=True,
        pre_norm=False,
        amp=True,
        # DataLoader performance
        num_workers=2,
        # === Gradient Accumulation ===
        gradient_accumulation_steps=2,
        # === Norman Continual Pretraining Config ===
        # Use full Norman data (100%) for maximum perturbation diversity
        use_norman_continual_pretrain=True,
        norman_epochs=10,
        norman_lr=2.5e-4,
        norman_batch_size=512,
        norman_n_hvg=896,
        norman_continue_from_pretrained=True,
        norman_subsample_ratio=1.0,
        norman_save_dir=str(PROJECT_ROOT / "save" / "scGPT_norman_continual_pretrain"),
        # Early stopping
        norman_patience=5,
        norman_min_delta=5e-5,
        norman_gradient_accumulation_steps=2,
        # Force reload PBMC to regenerate cache with gene symbols (critical fix)
        force_reload_pbmc=True,
    )

    # Try to use wandb; fall back to offline/no-wandb mode
    # Use wandb_logger (not 'wandb') to avoid shadowing the module-level import
    _wandb_logger = None
    if WANDB_AVAILABLE:
        try:
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(
                config=hyperparameter_defaults,
                project="scGPT",
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            config = wandb.config
            _wandb_logger = wandb  # use the real wandb module
            print("wandb initialized in OFFLINE mode")
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without wandb")
            config = SimpleConfig(hyperparameter_defaults)
            print(f"Running without wandb. Config: {config}")
            _wandb_logger = DummyWandb()
    else:
        config = SimpleConfig(hyperparameter_defaults)
        print(f"Running without wandb. Config: {config}")
        _wandb_logger = DummyWandb()

    # Make wandb_logger available globally for downstream references
    import builtins as _b
    _b.wandb_logger = _wandb_logger
    # Also alias so all references to 'wandb' in main() resolve correctly
    wandb_logger = _wandb_logger

    print(config)

    set_seed(config.seed)
    torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # Settings for input and preprocessing
    # -------------------------------------------------------------------------
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = -1
    pad_value = -2
    n_input_bins = config.n_bins

    # Use vocab-matched gene count as HVG target (vocab has ~896 gene symbols)
    # This ensures we keep most matched genes while still selecting the most variable ones
    n_hvg = max(getattr(config, 'norman_n_hvg', 896), 896)
    max_seq_len = n_hvg + 1
    per_seq_batch_sample = True
    DSBN = True
    explicit_zero_prob = True

    # -------------------------------------------------------------------------
    # Save directory setup
    # -------------------------------------------------------------------------
    dataset_name = config.dataset_name
    save_dir = Path(str(PROJECT_ROOT / "save" / f"dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}"))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    # -------------------------------------------------------------------------
    # Step 2: Load and pre-process data
    # -------------------------------------------------------------------------
    if dataset_name == "PBMC_10K":
        # Force reload PBMC data with gene symbols (critical fix: old cache used ENSG IDs
        # which only matched 7.9% of vocab; now using gene symbols for >80% match)
        force_reload = getattr(config, 'force_reload_pbmc', False)
        adata = load_pbmc_dataset(force_reload=force_reload)  # 11990 x ~3346 (gene symbols)
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        # var.index is already gene symbols (loaded with var_names='gene_symbols')
        # Only remap if the index is NOT already gene symbols (e.g., if cached data has ENSG IDs)
        if adata.var.index[0].startswith("ENSG"):
            logger.warning("var.index contains ENSG IDs, remapping to gene symbols...")
            if "gene_symbols" in adata.var.columns:
                adata.var = adata.var.set_index("gene_symbols")
            elif "gene_symbol" in adata.var.columns:
                adata.var = adata.var.set_index("gene_symbol")
            else:
                # Try to use the ensg_id column mapping
                logger.info("Gene symbols already in var.index, skipping remap.")
        data_is_raw = True

    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    # -------------------------------------------------------------------------
    # Load pretrained model vocabulary
    # -------------------------------------------------------------------------
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
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        embsize = config.layer_size
        nhead = config.nhead
        nlayers = config.nlayers
        d_hid = config.layer_size
        n_layers_cls = 1

    # -------------------------------------------------------------------------
    # Preprocess data
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Tokenize data
    # -------------------------------------------------------------------------
    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype"].tolist()
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

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

    # -------------------------------------------------------------------------
    # Step 3: Load the pre-trained scGPT model
    # -------------------------------------------------------------------------
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
    )
    if config.load_model is not None:
        load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)

    model.to(device)
    wandb_logger.watch(model)

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8,
        weight_decay=1e-4
    )
    # Cosine annealing LR scheduler with linear warmup for first 5 epochs
    warmup_epochs = min(5, config.epochs // 5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - warmup_epochs, eta_min=1e-6
    )
    # Use a simple warmup: for first warmup_epochs, linearly increase LR from 0.1*config.lr to config.lr
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / max(1, warmup_epochs - 1)
        else:
            return 1.0  # cosine scheduler handles remaining schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # Chain warmup -> cosine decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)

    # -------------------------------------------------------------------------
    # Stage 1: Continual Pretraining on Norman Perturb-seq Data
    # -------------------------------------------------------------------------
    if config.use_norman_continual_pretrain:
        logger.info("=" * 89)
        logger.info("Stage 1: Continual Pretraining on Norman Perturb-seq Data")
        logger.info("=" * 89)

        # 1. Load and preprocess Norman data
        norman_adata = load_norman_data(use_pertpy=True)
        norman_adata.var["gene_name"] = norman_adata.var.index.tolist()

        norman_adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in norman_adata.var["gene_name"]
        ]
        norman_gene_ids_in_vocab = np.array(norman_adata.var["id_in_vocab"])
        logger.info(
            f"Norman: match {np.sum(norman_gene_ids_in_vocab >= 0)}/{len(norman_gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        norman_adata = norman_adata[:, norman_adata.var["id_in_vocab"] >= 0]

        if config.norman_subsample_ratio < 1.0:
            n_cells = norman_adata.n_obs
            n_subsample = int(n_cells * config.norman_subsample_ratio)
            np.random.seed(config.seed)
            subsample_idx = np.random.choice(n_cells, n_subsample, replace=False)
            norman_adata = norman_adata[subsample_idx].copy()
            logger.info(
                f"Subsampled Norman data: {n_cells} -> {n_subsample} cells "
                f"({config.norman_subsample_ratio*100:.0f}%)"
            )

        data_max = norman_adata.X.max() if hasattr(norman_adata.X, 'max') else norman_adata.X.toarray().max()
        data_already_logged = data_max <= 30
        if data_already_logged:
            logger.info(f"Norman data appears already log1p-transformed (max={data_max:.2f}), skipping log1p.")
        norman_adata = preprocess_norman_data(
            norman_adata,
            n_hvg=config.norman_n_hvg,
            n_bins=config.n_bins,
            batch_key="guide_identity",
            data_is_raw=not data_already_logged,
        )

        norman_adata_sorted = norman_adata[
            norman_adata.obs["batch_id"].argsort()
        ].copy()

        norman_genes = norman_adata.var["gene_name"].tolist()
        norman_gene_ids = np.array(vocab(norman_genes), dtype=int)
        norman_batch_ids = norman_adata.obs["batch_id"].values
        norman_num_batch_types = len(set(norman_batch_ids))
        norman_max_seq_len = config.norman_n_hvg + 1

        norman_tokenized_train, norman_tokenized_valid, \
            norman_train_batch_labels, norman_valid_batch_labels = \
            prepare_norman_continual_pretrain_data(
                norman_adata,
                vocab,
                norman_gene_ids,
                max_seq_len=norman_max_seq_len,
                pad_token=pad_token,
                pad_value=pad_value,
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                test_size=0.1,
                random_state=config.seed,
            )

        norman_model = TransformerModel(
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
            num_batch_labels=norman_num_batch_types,
            domain_spec_batchnorm=DSBN,
            n_input_bins=n_input_bins,
            ecs_threshold=config.ecs_thres,
            explicit_zero_prob=explicit_zero_prob,
            use_fast_transformer=config.fast_transformer,
            pre_norm=config.pre_norm,
        )
        if config.load_model is not None and config.norman_continue_from_pretrained:
            load_pretrained(
                norman_model, torch.load(model_file, map_location=device), verbose=False
            )
        norman_model.to(device)

        norman_optimizer = torch.optim.AdamW(
            norman_model.parameters(), lr=config.norman_lr, eps=1e-4 if config.amp else 1e-8,
            weight_decay=1e-4
        )
        norman_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            norman_optimizer, T_max=config.norman_epochs, eta_min=1e-6
        )
        norman_scaler = torch.amp.GradScaler('cuda', enabled=config.amp)

        # ---- Inner functions for Norman stage ----
        def norman_prepare_data(sort_seq_batch=False):
            """Prepare Norman data for one epoch with dynamic masking."""
            masked_values_train = random_mask_value(
                norman_tokenized_train["values"],
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                pad_value=pad_value,
            )
            masked_values_valid = random_mask_value(
                norman_tokenized_valid["values"],
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                pad_value=pad_value,
            )

            input_gene_ids_train = norman_tokenized_train["genes"]
            input_values_train = masked_values_train
            target_values_train = norman_tokenized_train["values"]
            tensor_batch_labels_train = torch.from_numpy(
                norman_train_batch_labels
            ).long()

            input_gene_ids_valid = norman_tokenized_valid["genes"]
            input_values_valid = masked_values_valid
            target_values_valid = norman_tokenized_valid["values"]
            tensor_batch_labels_valid = torch.from_numpy(
                norman_valid_batch_labels
            ).long()

            if sort_seq_batch:
                train_sort_ids = np.argsort(norman_train_batch_labels)
                input_gene_ids_train = input_gene_ids_train[train_sort_ids]
                input_values_train = input_values_train[train_sort_ids]
                target_values_train = target_values_train[train_sort_ids]
                tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

                valid_sort_ids = np.argsort(norman_valid_batch_labels)
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

        def norman_train_epoch(loader, epoch_num):
            """Train Norman model for one epoch with gradient accumulation."""
            norman_model.train()
            total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
            total_error = 0.0
            log_interval = config.log_interval
            start_time = time.time()
            num_batches = len(loader)

            norman_accum_steps = getattr(config, 'norman_gradient_accumulation_steps',
                                         getattr(config, 'gradient_accumulation_steps', 1))
            norman_optimizer.zero_grad()

            for batch, batch_data in enumerate(loader):
                input_gene_ids = batch_data["gene_ids"].to(device, non_blocking=True)
                input_values = batch_data["values"].to(device, non_blocking=True)
                target_values = batch_data["target_values"].to(device, non_blocking=True)
                batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.amp.autocast('cuda', enabled=config.amp):
                    output_dict = norman_model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if DSBN else None,
                        MVC=config.GEPC,
                        ECS=config.ecs_thres > 0,
                    )

                    masked_positions = input_values.eq(mask_value)
                    loss = loss_mse = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    if explicit_zero_prob:
                        loss_zero_log_prob = criterion_neg_log_bernoulli(
                            output_dict["mlm_zero_probs"], target_values, masked_positions
                        )
                        loss = loss + loss_zero_log_prob
                    if config.GEPC:
                        loss_gepc = criterion(
                            output_dict["mvc_output"], target_values, masked_positions
                        )
                        loss = loss + loss_gepc
                    if config.GEPC and explicit_zero_prob:
                        loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                            output_dict["mvc_zero_probs"], target_values, masked_positions
                        )
                        loss = loss + loss_gepc_zero_log_prob
                    if config.ecs_thres > 0:
                        loss_ecs = 10 * output_dict["loss_ecs"]
                        loss = loss + loss_ecs
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + config.dab_weight * loss_dab

                loss = loss / norman_accum_steps
                norman_scaler.scale(loss).backward()

                if (batch + 1) % norman_accum_steps == 0 or (batch + 1) == num_batches:
                    norman_scaler.unscale_(norman_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        norman_model.parameters(),
                        1.0,
                        error_if_nonfinite=False if norman_scaler.is_enabled() else True,
                    )
                    norman_scaler.step(norman_optimizer)
                    norman_scaler.update()
                    norman_optimizer.zero_grad()

                with torch.no_grad():
                    mre = masked_relative_error(
                        output_dict["mlm_output"], target_values, masked_positions
                    )

                loss_item = loss.item() * norman_accum_steps
                total_loss += loss_item
                total_mse += loss_mse.item()
                total_gepc += loss_gepc.item() if config.GEPC else 0.0
                total_error += mre.item()
                if batch % log_interval == 0 and batch > 0:
                    lr = norman_scheduler.get_last_lr()[0]
                    ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                    cur_loss = total_loss / log_interval
                    cur_mse = total_mse / log_interval
                    cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
                    cur_error = total_error / log_interval
                    logger.info(
                        f"[Norman Stage] | epoch {epoch_num:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                        + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                    )
                    total_loss = 0
                    total_mse = 0
                    total_gepc = 0
                    total_error = 0
                    start_time = time.time()

        def norman_evaluate(loader):
            """Evaluate Norman model."""
            norman_model.eval()
            total_loss = 0.0
            total_error = 0.0
            total_dab = 0.0
            total_num = 0
            with torch.no_grad():
                for batch_data in loader:
                    input_gene_ids = batch_data["gene_ids"].to(device, non_blocking=True)
                    input_values = batch_data["values"].to(device, non_blocking=True)
                    target_values = batch_data["target_values"].to(device, non_blocking=True)
                    batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)

                    src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                    with torch.amp.autocast('cuda', enabled=config.amp):
                        output_dict = norman_model(
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

            return total_loss / total_num, total_error / total_num

        # ---- Norman Stage training loop with early stopping ----
        norman_best_val_loss = float("inf")
        norman_best_model = None
        norman_best_epoch = 0
        norman_save_dir = Path(config.norman_save_dir)
        norman_save_dir.mkdir(parents=True, exist_ok=True)

        norman_patience = getattr(config, 'norman_patience', 5)
        norman_min_delta = getattr(config, 'norman_min_delta', 5e-5)
        norman_no_improve_count = 0

        for norman_epoch in range(1, config.norman_epochs + 1):
            gc.collect()
            torch.cuda.empty_cache()

            norman_epoch_start = time.time()
            norman_train_pt, norman_valid_pt = norman_prepare_data(
                sort_seq_batch=False
            )
            # Use regular shuffled batching for Norman (no per-seq sampling needed
            # for continual pretraining - this is much faster with 290 batches)
            norman_train_loader = DataLoader(
                SeqDataset(norman_train_pt),
                batch_size=config.norman_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,  # Use 0 workers to avoid spawn issues with inner functions
                pin_memory=True,
            )
            norman_valid_loader = DataLoader(
                SeqDataset(norman_valid_pt),
                batch_size=config.norman_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,  # Use 0 workers to avoid spawn issues with inner functions
                pin_memory=True,
            )

            norman_train_epoch(norman_train_loader, norman_epoch)
            norman_val_loss, norman_val_mre = norman_evaluate(norman_valid_loader)
            norman_elapsed = time.time() - norman_epoch_start
            logger.info("-" * 89)
            logger.info(
                f"[Norman Stage] | end of epoch {norman_epoch:3d} | "
                f"time: {norman_elapsed:5.2f}s | "
                f"valid loss/mse {norman_val_loss:5.4f} | mre {norman_val_mre:5.4f}"
            )
            logger.info("-" * 89)

            if norman_val_loss < norman_best_val_loss - norman_min_delta:
                norman_best_val_loss = norman_val_loss
                norman_best_model = copy.deepcopy(norman_model)
                norman_best_epoch = norman_epoch
                norman_no_improve_count = 0
                logger.info(
                    f"[Norman Stage] Best model with score {norman_best_val_loss:5.4f}"
                )
            else:
                norman_no_improve_count += 1
                logger.info(
                    f"[Norman Stage] No improvement for {norman_no_improve_count} epoch(s)."
                )
                if norman_no_improve_count >= norman_patience:
                    logger.info(
                        f"[Norman Stage] Early stopping triggered after {norman_epoch} epochs."
                    )
                    break

            norman_scheduler.step()

            del norman_train_pt, norman_valid_pt, norman_train_loader, norman_valid_loader
            gc.collect()
            torch.cuda.empty_cache()

        # Save Norman Stage 1 checkpoint
        norman_ckpt_path = norman_save_dir / "best_model.pt"
        torch.save(norman_best_model.state_dict(), norman_ckpt_path)
        logger.info(f"[Norman Stage] Checkpoint saved to {norman_ckpt_path}")

        norman_args = {
            "embsize": embsize,
            "nheads": nhead,
            "d_hid": d_hid,
            "nlayers": nlayers,
            "n_layers_cls": n_layers_cls,
        }
        with open(norman_save_dir / "args.json", "w") as f:
            json.dump(norman_args, f)
        if hasattr(vocab, "save_json"):
            vocab.save_json(norman_save_dir / "vocab.json")
        else:
            import json as _json
            _json.dump(vocab.get_stoi(), open(norman_save_dir / "vocab.json", "w"), indent=2)

        logger.info("=" * 89)
        logger.info("Stage 1 Complete. Starting Stage 2: PBMC 10K Integration Fine-tuning")
        logger.info("=" * 89)

        # Load Stage 1 checkpoint as initialization for Stage 2
        load_pretrained(
            model,
            torch.load(norman_ckpt_path, map_location=device),
            verbose=False,
        )
        model.to(device)
        logger.info("Loaded Norman continual pretrained weights for Stage 2.")

        # Cleanup Norman stage memory
        # Note: per-epoch variables (norman_train_pt, norman_valid_pt, etc.) may already
        # be deleted inside the loop's last iteration before reaching here.
        del norman_model, norman_optimizer, norman_scheduler, norman_scaler
        del norman_tokenized_train, norman_tokenized_valid
        del norman_train_batch_labels, norman_valid_batch_labels
        try:
            del norman_train_pt, norman_valid_pt, norman_train_loader, norman_valid_loader
        except (UnboundLocalError, NameError):
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Stage 2: Standard PBMC 10K Integration Fine-tuning
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    best_model_epoch = None
    best_avg_bio_model = None
    best_avg_bio_epoch = 0
    define_wandb_metrcis(wandb_logger)

    for epoch in range(1, config.epochs + 1):
        gc.collect()
        torch.cuda.empty_cache()

        # Cosine mask ratio schedule: start near mask_ratio, decay to ~0.08
        # High mask ratio early: expose model to more diverse contexts for better learning
        # Low mask ratio late: fine-grained tuning with more unmasked context
        # Floor at 0.08 ensures sufficient learning signal throughout
        mask_ratio_schedule = max(
            mask_ratio * (1 + np.cos(np.pi * epoch / config.epochs)) / 2,
            0.08  # minimum mask ratio for sustained learning signal
        )

        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(
            sort_seq_batch=per_seq_batch_sample,
            epoch=epoch,
            mask_ratio_schedule=mask_ratio_schedule,
            tokenized_train=tokenized_train,
            tokenized_valid=tokenized_valid,
            train_batch_labels=train_batch_labels,
            valid_batch_labels=valid_batch_labels,
            mask_value=mask_value,
            pad_value=pad_value,
            mask_ratio=mask_ratio,
        )
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
            config=config,
            num_workers=0,  # Use 0 workers to avoid DataLoader spawn/multiprocessing issues
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
            per_seq_batch_sample=per_seq_batch_sample,
            config=config,
            num_workers=0,  # Use 0 workers to avoid DataLoader spawn/multiprocessing issues
        )

        if config.do_train:
            train(
                model=model, loader=train_loader, config=config, device=device,
                vocab=vocab, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, criterion=criterion, criterion_dab=criterion_dab,
                mask_value=mask_value, pad_token=pad_token, epoch=epoch,
                mask_ratio=mask_ratio, logger=logger, wandb=wandb_logger,
            )
        val_loss, val_mre = evaluate(
            model=model, loader=valid_loader, config=config, device=device,
            vocab=vocab, criterion=criterion, criterion_dab=criterion_dab,
            mask_value=mask_value, pad_token=pad_token, epoch=epoch, wandb=wandb_logger,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        # Track best model by validation loss, but also log avg_bio for final selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")
        # Save best avg_bio model for final evaluation metric optimization
        if epoch == 1:
            best_avg_bio_model = copy.deepcopy(model)
            best_avg_bio_epoch = 1

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            # Use the best model for evaluation
            eval_model = best_model if best_model is not None else model
            torch.save(eval_model.state_dict(), save_dir / f"model_e{best_model_epoch if best_model is not None else epoch}.pt")

            results = eval_testdata(
                eval_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                config=config,
                device=device,
                vocab=vocab,
                input_layer_key=input_layer_key,
                gene_ids=gene_ids,
                max_seq_len=max_seq_len,
                pad_token=pad_token,
                pad_value=pad_value,
                DSBN=DSBN,
                logger=logger,
                include_types=["cls"],
            )
            if results:
                current_avg_bio = results.get('avg_bio', 0.0)
                if current_avg_bio > best_avg_bio:
                    best_avg_bio = current_avg_bio
                    best_avg_bio_model = copy.deepcopy(model)
                    best_avg_bio_epoch = epoch
                    logger.info(f"New best avg_bio: {best_avg_bio:.4f} at epoch {epoch}")
                results["batch_umap"].savefig(
                    save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
                )
                results["celltype_umap"].savefig(
                    save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
                )
                metrics_to_log = {"test/" + k: v for k, v in results.items()}
                wandb_logger.log(metrics_to_log)
                wandb_logger.log({"avg_bio": results.get("avg_bio", 0.0)})

        scheduler.step()

        del train_data_pt, valid_data_pt, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Save best model (by val_loss) and best avg_bio model
    # -------------------------------------------------------------------------
    # Save best model by validation loss
    if best_model is not None:
        try:
            best_model_path = save_dir / "best_model.pt"
            torch.save(best_model.state_dict(), best_model_path)
            logger.info(f"Best model (by val_loss) saved to {best_model_path}")
            if best_model_epoch is not None:
                epoch_path = save_dir / f"best_model_e{best_model_epoch}.pt"
                torch.save(best_model.state_dict(), epoch_path)
                logger.info(f"Best model (epoch {best_model_epoch}) saved to {epoch_path}")
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
    else:
        logger.warning("No best model to save!")
        try:
            torch.save(model.state_dict(), save_dir / "final_model.pt")
            logger.info(f"Current model saved to {save_dir / 'final_model.pt'}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

    # Save best model by avg_bio (primary evaluation metric)
    if best_avg_bio > 0:
        try:
            avg_bio_path = save_dir / "best_avg_bio_model.pt"
            torch.save(best_avg_bio_model.state_dict(), avg_bio_path)
            logger.info(f"Best avg_bio model ({best_avg_bio:.4f}, epoch {best_avg_bio_epoch}) saved to {avg_bio_path}")
        except Exception as e:
            logger.error(f"Failed to save best avg_bio model: {e}")

    # Final evaluation on the best avg_bio model
    if best_avg_bio > 0:
        try:
            logger.info(f"Running final evaluation with best avg_bio model (epoch {best_avg_bio_epoch}, avg_bio={best_avg_bio:.4f})...")
            final_results = eval_testdata(
                best_avg_bio_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                config=config,
                device=device,
                vocab=vocab,
                input_layer_key=input_layer_key,
                gene_ids=gene_ids,
                max_seq_len=max_seq_len,
                pad_token=pad_token,
                pad_value=pad_value,
                DSBN=DSBN,
                logger=logger,
                include_types=["cls"],
            )
            if final_results:
                final_results["batch_umap"].savefig(
                    save_dir / "final_embeddings_batch_umap[cls].png", dpi=300
                )
                final_results["celltype_umap"].savefig(
                    save_dir / "final_embeddings_celltype_umap[cls].png", dpi=300
                )
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    if WANDB_AVAILABLE and 'run' in dir():
        try:
            artifact = wandb_logger.Artifact("best_model", type="model")
            glob_str = os.path.join(save_dir, "best_model.pt")
            artifact.add_file(glob_str)
            run.log_artifact(artifact)
            run.finish()
            wandb_logger.finish()
        except Exception as e:
            print(f"wandb finish warning: {e}")
    gc.collect()

    logger.info("=" * 89)
    logger.info("Training pipeline complete!")
    logger.info("=" * 89)


# =============================================================================
# Entry point: only runs in the main process, NOT in DataLoader workers
# =============================================================================
if __name__ == "__main__":
    main()