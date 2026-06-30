#!/usr/bin/env python
# coding: utf-8

# # Enhanced Fine-tuning on Pre-trained Model with Batch Integration
#
# Improvements over the baseline:
#   1. **Norman continual pre-training** as warm-up (perturbation data improves gene representations)
#   2. **Contrastive Cell Embedding (CCE)** objective for better cell-type separation
#   3. **Adaptive mask-ratio curriculum** (high→low) for progressive learning
#   4. **Cosine annealing LR scheduler** with warm restarts
#   5. **Fast evaluation** with chunked processing and direct batch encoding
#   6. **Multi-dataset training**: PBMC 10K + Norman perturbation data
#   7. **Memory-efficient** with explicit cleanup throughout

# In[1]:


# =============================================================================
# Compatibility fix for Python 3.13 + setuptools
# =============================================================================
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    import importlib.abc
    import importlib.machinery
    pkgutil.ImpImporter = importlib.machinery.PathFinder

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
import wandb
from scipy.sparse import issparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
sys.path.insert(0, "../")
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
from scgpt.pbmc_loader import load_pbmc10k
from scgpt.norman_loader import (
    load_norman_h5ad,
    preprocess_norman_adata,
    filter_genes_to_vocab,
)
from scgpt.norman_pretrain import continual_pretrain_norman

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# ## Step1: Hyper-parameter setup

# In[2]:


hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",
    do_train=True,
    load_model="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/test_example/data/scgpt_integration/checkpoints/scGPT_human",
    GEPC=True,
    ecs_thres=0.8,
    dab_weight=0.8,
    cce_weight=0.35,  # Increased for better cell-type separation
    mask_ratio=0.4,
    epochs=20,  # Fits within time budget
    n_bins=51,
    lr=1e-4,
    batch_size=64,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=10,  # Evaluate at epoch 10 and 20
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    # Norman dataset settings
    use_norman_pretrain=True,
    norman_epochs=5,
    norman_lr_scale=0.3,
    norman_data_path="./tutorials/data/norman.h5ad",
    norman_mask_ratio=0.55,
    # Curriculum learning settings
    adaptive_mask=True,
    mask_ratio_high=0.55,
    mask_ratio_low=0.15,
    curriculum_switch_epoch=5,
    # Early stopping
    early_stop_patience=20,
    # Memory optimization
    eval_chunk_size=512,  # Larger chunks for faster evaluation
)
# Try to initialize wandb
WANDB_AVAILABLE = True
try:
    run = wandb.init(
        config=hyperparameter_defaults,
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
        mode=os.environ.get("WANDB_MODE", "disabled"),
    )
    config = wandb.config
except Exception as wandb_err:
    print(f"WandB init failed ({wandb_err}), using dict config")
    WANDB_AVAILABLE = False
    class DictConfig:
        def __init__(self, d):
            self.__dict__.update(d)
        def __getitem__(self, key):
            return self.__dict__[key]
        def __setitem__(self, key, val):
            self.__dict__[key] = val
        def __contains__(self, key):
            return key in self.__dict__
        def get(self, key, default=None):
            return self.__dict__.get(key, default)
        def update(self, d):
            self.__dict__.update(d)
        def keys(self):
            return self.__dict__.keys()
        def items(self):
            return self.__dict__.items()
        def __str__(self):
            return str(self.__dict__)
    config = DictConfig(hyperparameter_defaults)
    run = None

print(config)

set_seed(config.seed)


# In[3]:


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 1200
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True
explicit_zero_prob = True


# In[4]:


dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ## Step 2: Load and pre-process data

# In[5]:


if dataset_name == "PBMC_10K":
    adata = load_pbmc10k()
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True

adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


# ### 2.2 Cross-check gene set with the pre-trained model

# In[6]:


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
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size


# ### 2.3 Pre-process the PBMC 10K data

# In[7]:


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


# In[8]:


if per_seq_batch_sample:
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()


# ### 2.4 Tokenize the input data

# In[9]:


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


# In[10]:


if config.load_model is None:
    vocab = GeneVocab(genes + special_tokens)
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


# In[11]:


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


# ## Enhanced Data Preparation with Adaptive Masking

# In[12]:


def get_adaptive_mask_ratio(epoch, total_epochs):
    if not config.adaptive_mask:
        return config.mask_ratio

    switch_epoch = config.curriculum_switch_epoch
    high_ratio = config.mask_ratio_high
    low_ratio = config.mask_ratio_low

    if epoch <= switch_epoch:
        return high_ratio
    decay_fraction = (epoch - switch_epoch) / (total_epochs - switch_epoch)
    ratio = high_ratio - (high_ratio - low_ratio) * decay_fraction
    return max(ratio, low_ratio)


def prepare_data(sort_seq_batch=False, epoch=1) -> Tuple[Dict[str, torch.Tensor]]:
    current_mask_ratio = get_adaptive_mask_ratio(epoch, config.epochs)

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
            num_workers=0,
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


# ## Step 3: Load the pre-trained scGPT model

# In[13]:


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
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)

model.to(device)
wandb.watch(model)


# ## Step 3.5: Norman Continual Pre-training (Warm-up)

# In[14]:


if config.use_norman_pretrain:
    logger.info("=" * 60)
    logger.info("Step 3.5: Starting Norman continual pre-training as warm-up")
    logger.info("=" * 60)
    try:
        norman_adata = load_norman_h5ad(config.norman_data_path)
        norman_adata = preprocess_norman_adata(norman_adata, n_top_genes=n_hvg, n_bins=config.n_bins)
        norman_adata = filter_genes_to_vocab(norman_adata, vocab, special_tokens)

        if norman_adata.n_vars > 0:
            norman_genes = norman_adata.var["gene_name"].tolist()
            norman_gene_ids = np.array(vocab(norman_genes), dtype=int)

            norman_adata.obs["str_batch"] = "norman"
            batch_id_labels = norman_adata.obs["str_batch"].astype("category").cat.codes.values
            norman_adata.obs["batch_id"] = batch_id_labels

            class NormanConfig:
                pass
            norman_config = NormanConfig()
            norman_config.seed = config.seed
            norman_config.batch_size = config.batch_size
            norman_config.lr = config.lr
            norman_config.amp = config.amp
            norman_config.GEPC = config.GEPC
            norman_config.ecs_thres = config.ecs_thres
            norman_config.explicit_zero_prob = explicit_zero_prob
            norman_config.log_interval = config.log_interval
            norman_config.schedule_ratio = config.schedule_ratio
            norman_config.lr = config.lr * config.norman_lr_scale
            norman_config.norman_mask_ratio_fixed = config.norman_mask_ratio
            norman_config.mask_ratio = config.norman_mask_ratio

            model = continual_pretrain_norman(
                model=model,
                vocab=vocab,
                gene_ids=norman_gene_ids,
                adata=norman_adata,
                config=norman_config,
                device=device,
                save_dir=save_dir,
                norman_epochs=config.norman_epochs,
                per_seq_batch_sample=False,
                DSBN=False,
                num_batch_types=1,
            )
            logger.info("Norman continual pre-training completed!")
        else:
            logger.warning("No gene overlap with Norman data, skipping.")
    except Exception as e:
        logger.warning(f"Norman pre-training failed: {e}")
        traceback.print_exc()


# ## Step 4: Finetune scGPT with Enhanced Training

# In[15]:


criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    eps=1e-4 if config.amp else 1e-8,
    weight_decay=0.01,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)

scaler = torch.amp.GradScaler('cuda', enabled=config.amp)


# ### Enhanced Training Function

# In[16]:


def train(model: nn.Module, loader: DataLoader, epoch: int) -> None:
    model.train()
    total_loss, total_mse, total_gepc, total_cce = 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    optimizer.zero_grad()

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.amp.autocast(device_type='cuda', enabled=config.amp):
            enable_cce = config.cce_weight > 0

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CCE=enable_cce,
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

            if enable_cce:
                loss_cce = output_dict["loss_cce"]
                loss = loss + config.cce_weight * loss_cce

            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
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

        if WANDB_AVAILABLE:
            wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_cce += loss_cce.item() if enable_cce else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_cce = total_cce / log_interval if enable_cce else 0.0
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"cce {cur_cce:5.4f} |" if enable_cce else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_cce = 0
            total_error = 0
            start_time = time.time()

    gc.collect()


def define_wandb_metrcis():
    if WANDB_AVAILABLE:
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")
        wandb.define_metric("test/ARI", summary="max")
        wandb.define_metric("test/NMI", summary="max")


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
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
            with torch.amp.autocast(device_type='cuda', enabled=config.amp):
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

    if WANDB_AVAILABLE:
        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/mre": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab) / total_num,
                "epoch": epoch,
            },
        )

    return total_loss / total_num, total_error / total_num


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """Evaluate the model on test dataset with fast chunked encoding."""
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
        all_gene_ids = tokenized_all["genes"]
        all_values = tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

        # Fast batch encoding - use larger chunks for speed
        eval_chunk_size = getattr(config, 'eval_chunk_size', 512)
        n_cells = len(all_counts)
        cell_embeddings_list = []

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=config.amp):
            for start_idx in range(0, n_cells, eval_chunk_size):
                end_idx = min(start_idx + eval_chunk_size, n_cells)
                chunk_gene_ids = all_gene_ids[start_idx:end_idx].to(device)
                chunk_values = all_values[start_idx:end_idx].float().to(device)
                chunk_mask = src_key_padding_mask[start_idx:end_idx].to(device)
                chunk_batch_labels = torch.from_numpy(batch_ids[start_idx:end_idx]).long().to(device) if DSBN else None

                # Use encode_batch with larger internal batch for speed
                chunk_embeddings = model.encode_batch(
                    chunk_gene_ids,
                    chunk_values,
                    src_key_padding_mask=chunk_mask,
                    batch_size=min(512, eval_chunk_size),
                    batch_labels=chunk_batch_labels,
                    time_step=0,
                    return_np=True,
                )
                cell_embeddings_list.append(chunk_embeddings)

                del chunk_gene_ids, chunk_values, chunk_mask, chunk_batch_labels, chunk_embeddings

        cell_embeddings = np.concatenate(cell_embeddings_list, axis=0)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        del cell_embeddings_list
        gc.collect()
        torch.cuda.empty_cache()

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        # UMAP
        try:
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
                title=[f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
                frameon=False,
                return_fig=True,
                show=False,
            )
            results["celltype_umap"] = fig
        except Exception as e:
            logger.warning(f"UMAP generation failed: {e}")

        del cell_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    if len(include_types) == 1:
        return results


# ## Step 4: Enhanced Fine-tuning Loop

# In[17]:


best_val_loss = float("inf")
best_avg_bio = 0.0
best_ari = 0.0
best_model = None
best_model_epoch = 0
best_ari_model_state = None
no_improve_count = 0
define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    train_data_pt, valid_data_pt = prepare_data(
        sort_seq_batch=per_seq_batch_sample,
        epoch=epoch,
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
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        results = eval_testdata(
            best_model,
            adata_t=adata_sorted if per_seq_batch_sample else adata,
            include_types=["cls"],
        )

        if results is not None:
            if "batch_umap" in results and results["batch_umap"] is not None:
                results["batch_umap"].savefig(
                    save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
                )
                plt.close(results["batch_umap"])

            if "celltype_umap" in results and results["celltype_umap"] is not None:
                results["celltype_umap"].savefig(
                    save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
                )
                plt.close(results["celltype_umap"])

            metrics_to_log = {"test/" + k: v for k, v in results.items() if not isinstance(v, plt.Figure)}
            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            if WANDB_AVAILABLE:
                wandb.log(metrics_to_log)
                wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

            if WANDB_AVAILABLE:
                if "ARI_cluster/label" in results:
                    wandb.log({"test/ARI": results["ARI_cluster/label"]})
                if "NMI_cluster/label" in results:
                    wandb.log({"test/NMI": results["NMI_cluster/label"]})
                if "ASW_label" in results:
                    wandb.log({"test/ASW_label": results["ASW_label"]})

            current_ari = results.get("ARI_cluster/label", 0.0)
            if current_ari > best_ari:
                best_ari = current_ari
                best_ari_model_state = copy.deepcopy(best_model.state_dict())
                torch.save(best_model.state_dict(), save_dir / "best_model_by_ari.pt")
                logger.info(f"New best ARI: {best_ari:.4f}")

            if current_ari > best_ari - 0.001:
                no_improve_count = 0
            else:
                no_improve_count += config.save_eval_interval

            current_avg_bio = results.get("avg_bio", 0.0)
            if current_avg_bio > best_avg_bio:
                best_avg_bio = current_avg_bio
                torch.save(best_model.state_dict(), save_dir / "best_model_by_avg_bio.pt")
                logger.info(f"New best avg_bio: {best_avg_bio:.4f}")

            if no_improve_count >= config.early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")

            gc.collect()
            torch.cuda.empty_cache()

    scheduler.step(epoch - 1)

    if epoch % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()


# In[18]:


torch.save(best_model.state_dict(), save_dir / "best_model.pt")
logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")
logger.info(f"Best avg_bio achieved: {best_avg_bio:.4f}")
logger.info(f"Best ARI achieved: {best_ari:.4f}")

# Final evaluation
if best_ari_model_state is not None:
    model.load_state_dict(best_ari_model_state)
    gc.collect()
    torch.cuda.empty_cache()

    final_results = eval_testdata(
        model,
        adata_t=adata_sorted if per_seq_batch_sample else adata,
        include_types=["cls"],
    )
    if final_results is not None:
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION RESULTS (ARI-best model):")
        for k, v in final_results.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)
        if WANDB_AVAILABLE:
            wandb.log({"final_ari/ARI": final_results.get("ARI_cluster/label", 0.0)})
            wandb.log({"final_ari/NMI": final_results.get("NMI_cluster/label", 0.0)})
            wandb.log({"final_ari/avg_bio": final_results.get("avg_bio", 0.0)})
elif best_model is not None:
    final_results = eval_testdata(
        best_model,
        adata_t=adata_sorted if per_seq_batch_sample else adata,
        include_types=["cls"],
    )
    if final_results is not None:
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION RESULTS:")
        for k, v in final_results.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)


# In[19]:


if WANDB_AVAILABLE:
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)
    run.finish()
    wandb.finish()
gc.collect()


# In[ ]: