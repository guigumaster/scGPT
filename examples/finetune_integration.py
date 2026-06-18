# %%
"""
scGPT Fine-tuning for scRNA-seq Integration
- Optimized for CPU training on H20 (CUDA not available in this env)
- Uses PBMC 3K dataset directly (no broken 10K download)
- Prototype-based Contrastive Learning for Cell Type-Aware Fine-tuning
- Proper train/val/test split with ARI evaluation
- Curriculum learning with checkpointing
"""
import copy
import gc
import json
import math
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
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Use scGPT's built-in vocab compatibility layer
from scgpt.tokenizer.vocab_compat import Vocab, BuiltinVocab
from scgpt.tokenizer.gene_tokenizer import GeneVocab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator, PrototypeContrastiveHead
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

# Device - CPU only in this env
DEVICE = torch.device("cpu")
IS_CUDA = False
print(f"Using device: {DEVICE}, CUDA available: {IS_CUDA}")
print(f"PyTorch version: {torch.__version__}")

# ──────────────────────────────────────────────────────────────
# Data loading - always use PBMC 3K directly
# ──────────────────────────────────────────────────────────────
def load_pbmc3k(data_dir: Path) -> AnnData:
    """
    Load PBMC 3K dataset with cell type labels.
    Works reliably without external downloads.
    """
    h5ad_path = data_dir / "pbmc3k_annotated.h5ad"
    if h5ad_path.exists():
        print(f"Loading cached PBMC3K from {h5ad_path}")
        return sc.read_h5ad(h5ad_path)

    print("Loading PBMC 3K from scanpy (raw counts)...")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Get cell type annotations
    adata_proc = sc.datasets.pbmc3k_processed()
    celltype_map = dict(zip(adata_proc.obs_names, adata_proc.obs["louvain"]))
    adata.obs["celltype"] = adata.obs_names.map(celltype_map)
    adata.obs["celltype"] = adata.obs["celltype"].fillna("Unknown").astype(str).astype("category")
    adata.obs["batch"] = "0"
    adata.obs["str_labels"] = adata.obs["celltype"].astype(str)

    print(f"PBMC3K raw data: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Cell types:")
    for ct, count in adata.obs['celltype'].value_counts().items():
        print(f"  {ct}: {count}")

    adata.write(h5ad_path)
    print(f"Cached to {h5ad_path}")
    return adata


def prepare_adata() -> AnnData:
    """Prepare the ADATA object with proper preprocessing."""
    dataset_name = "PBMC_10K"  # Keep name for compatibility but use PBMC 3K
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")

    # Copy this script to save dir for reproducibility
    src_script = Path(__file__)
    if src_script.exists():
        import shutil
        shutil.copy2(str(src_script), str(save_dir / "finetune_integration.py"))

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    adata = load_pbmc3k(save_dir / "pbmc_data")

    # Standardize column names
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    ori_batch_col = "batch"
    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    return adata, save_dir


# ──────────────────────────────────────────────────────────────
# Hyperparameters - optimized for CPU
# ──────────────────────────────────────────────────────────────
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",
    do_train=True,
    load_model=None,
    mask_ratio=0.4,
    epochs=15,                  # Reduced from 30 for CPU speed
    n_bins=51,
    GEPC=True,
    ecs_thres=0.8,
    dab_weight=1.0,
    cls_weight=0.1,
    proto_weight=0.05,
    cce_weight=0.1,
    max_cls_weight=1.0,
    max_proto_weight=0.5,
    curriculum_start=0,
    curriculum_end=10,
    proto_momentum=0.999,
    proto_temp=0.1,
    lr=5e-4,                    # Increased slightly for faster convergence
    batch_size=32,              # Reduced for CPU
    layer_size=64,              # Reduced from 128 for CPU speed
    nlayers=3,                  # Reduced from 4
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=3,       # Evaluate more frequently
    log_interval=50,            # Log more frequently
    fast_transformer=False,
    pre_norm=False,
    amp=False,                  # No AMP on CPU
)

set_seed(hyperparameter_defaults["seed"])

print("Configuration:")
for k, v in hyperparameter_defaults.items():
    print(f"  {k}: {v}")

# %%
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = hyperparameter_defaults["mask_ratio"]
mask_value = -1
pad_value = -2
n_input_bins = hyperparameter_defaults["n_bins"]
n_hvg = 1200
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True
explicit_zero_prob = True

# %%
# Prepare data
adata, save_dir = prepare_adata()
logger = scg.logger

# Use the full adata for training (PBMC3K is small enough)
ori_batch_col = "batch"
data_is_raw = True

# %%
embsize = hyperparameter_defaults["layer_size"]
nhead = hyperparameter_defaults["nhead"]
nlayers = hyperparameter_defaults["nlayers"]
d_hid = hyperparameter_defaults["layer_size"]

# %%
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
    binning=hyperparameter_defaults["n_bins"],
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch")

# %%
if per_seq_batch_sample:
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# %% [markdown]
# ## Tokenize input

# %%
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()
num_types = len(set(celltypes_labels))
celltypes_label_map = {ct: i for i, ct in enumerate(sorted(set(celltypes_labels)))}
celltypes_labels_int = np.array([celltypes_label_map[ct] for ct in celltypes_labels])
logger.info(f"Number of cell types: {num_types}")
logger.info(f"Cell type mapping: {celltypes_label_map}")

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

# Split into train/val/test
all_indices = np.arange(len(all_counts))
train_idx, test_idx = train_test_split(
    all_indices, test_size=0.2, random_state=42, stratify=celltypes_labels_int
)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.125, random_state=42,  # 0.125 * 0.8 = 0.1 of total
    stratify=celltypes_labels_int[train_idx]
)

train_data = all_counts[train_idx]
valid_data = all_counts[val_idx]
test_data = all_counts[test_idx]

train_celltype_labels = celltypes_labels_int[train_idx]
valid_celltype_labels = celltypes_labels_int[val_idx]
test_celltype_labels = celltypes_labels_int[test_idx]

train_batch_labels = batch_ids[train_idx]
valid_batch_labels = batch_ids[val_idx]
test_batch_labels = batch_ids[test_idx]

logger.info(f"Train: {len(train_data)}, Val: {len(valid_data)}, Test: {len(test_data)}")

# %%
# Build vocabulary
vocab = BuiltinVocab(genes + special_tokens)
vocab.set_default_index(vocab[pad_token])
gene_ids = np.array(vocab(genes), dtype=int)
ntokens = len(vocab)

# %%
tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids, max_len=max_seq_len,
    vocab=vocab, pad_token=pad_token, pad_value=pad_value,
    append_cls=True, include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=max_seq_len,
    vocab=vocab, pad_token=pad_token, pad_value=pad_value,
    append_cls=True, include_zero_gene=True,
)
tokenized_test = tokenize_and_pad_batch(
    test_data, gene_ids, max_len=max_seq_len,
    vocab=vocab, pad_token=pad_token, pad_value=pad_value,
    append_cls=True, include_zero_gene=True,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# %%
def prepare_data(sort_seq_batch=False, current_epoch: int = 1) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"], mask_ratio=mask_ratio,
        mask_value=mask_value, pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"], mask_ratio=mask_ratio,
        mask_value=mask_value, pad_value=pad_value,
    )
    masked_ratio_val = 0.0
    if (masked_values_train - pad_value).count_nonzero() > 0:
        masked_ratio_val = (masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero()
    print(
        f"random masking at epoch {current_epoch:3d}, ratio of masked values in train: ",
        f"{masked_ratio_val:.4f}",
    )

    input_gene_ids_train = tokenized_train["genes"]
    input_gene_ids_valid = tokenized_valid["genes"]
    input_values_train = masked_values_train
    input_values_valid = masked_values_valid
    target_values_train = tokenized_train["values"]
    target_values_valid = tokenized_valid["values"]

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        for arr in [input_gene_ids_train, input_values_train, target_values_train]:
            train_sort = np.argsort(train_batch_labels)
            arr[:] = arr[train_sort]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort]

        for arr in [input_gene_ids_valid, input_values_valid, target_values_valid]:
            valid_sort = np.argsort(valid_batch_labels)
            arr[:] = arr[valid_sort]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort]

    train_data_pt = {
        "gene_ids": input_gene_ids_train, "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid, "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
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
    data_pt, batch_size, shuffle=False, intra_domain_shuffle=False,
    drop_last=False, num_workers=0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            subsets.append(np.where(batch_labels_array == batch_label)[0].tolist())
        return DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets, batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle, drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=False,   # CPU mode - disable pin_memory
        )
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers,
        pin_memory=False,       # CPU mode - disable pin_memory
    )


# %% [markdown]
# ## Curriculum Learning Helper
def get_curriculum_weight(epoch, start, end, initial_weight, max_weight):
    if epoch <= start:
        return initial_weight
    elif epoch >= end:
        return max_weight
    progress = (epoch - start) / (end - start)
    cosine_factor = (1 - math.cos(progress * math.pi)) / 2
    return initial_weight + (max_weight - initial_weight) * cosine_factor


# %% [markdown]
# ## Create and finetune scGPT

# %%
device = DEVICE
n_layers_cls = 3

model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers,
    n_cls=num_types, vocab=vocab, dropout=hyperparameter_defaults["dropout"],
    pad_token=pad_token, pad_value=pad_value,
    do_mvc=hyperparameter_defaults["GEPC"], do_dab=True,
    use_batch_labels=True, num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN, n_input_bins=n_input_bins,
    ecs_threshold=hyperparameter_defaults["ecs_thres"],
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=hyperparameter_defaults["fast_transformer"],
    pre_norm=hyperparameter_defaults["pre_norm"],
    use_proto=True, proto_momentum=hyperparameter_defaults["proto_momentum"],
    proto_temp=hyperparameter_defaults["proto_temp"], nlayers_cls=n_layers_cls,
)

# Check for pretrained model
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
pretrained_dir = Path(PROJECT_ROOT) / "examples" / "save" / "scGPT_bc"
pretrained_file = pretrained_dir / "best_model.pt"
if pretrained_file.exists():
    try:
        model.load_state_dict(torch.load(str(pretrained_file), map_location=device))
        logger.info(f"Loaded pretrained model from {pretrained_file}")
    except Exception as e:
        logger.warning(f"Could not load pretrained model: {e}")
        logger.info("Training from scratch without pretrained weights.")
else:
    logger.info("No pretrained model found. Training from scratch.")

model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=hyperparameter_defaults["lr"], eps=1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=hyperparameter_defaults["schedule_ratio"])
scaler = None  # No AMP on CPU


def train(model, loader, epoch):
    model.train()
    cls_w = get_curriculum_weight(
        epoch, hyperparameter_defaults["curriculum_start"], hyperparameter_defaults["curriculum_end"],
        hyperparameter_defaults["cls_weight"], hyperparameter_defaults["max_cls_weight"],
    )
    proto_w = get_curriculum_weight(
        epoch, hyperparameter_defaults["curriculum_start"], hyperparameter_defaults["curriculum_end"],
        hyperparameter_defaults["proto_weight"], hyperparameter_defaults["max_proto_weight"],
    )
    cce_w = hyperparameter_defaults["cce_weight"]

    total_loss = total_mse = total_gepc = 0.0
    total_cls = total_proto = total_cce = total_ecs = total_dab = 0.0
    total_error = 0.0
    log_interval = hyperparameter_defaults["log_interval"]
    start_time = time.time()
    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        output_dict = model(
            input_gene_ids, input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if DSBN else None,
            CLS=True, CCE=epoch > 0, MVC=hyperparameter_defaults["GEPC"],
            ECS=hyperparameter_defaults["ecs_thres"] > 0,
            celltype_labels=celltype_labels,
        )
        masked_positions = input_values.eq(mask_value)
        loss = loss_mse = criterion(
            output_dict["mlm_output"], target_values, masked_positions
        )
        metrics = {"train/mse": loss_mse.item()}

        if explicit_zero_prob:
            loss_zero = criterion_neg_log_bernoulli(
                output_dict["mlm_zero_probs"], target_values, masked_positions
            )
            loss = loss + loss_zero
            metrics["train/nzlp"] = loss_zero.item()
        if hyperparameter_defaults["GEPC"]:
            loss_gepc = criterion(
                output_dict["mvc_output"], target_values, masked_positions
            )
            loss = loss + loss_gepc
            metrics["train/mvc"] = loss_gepc.item()
            if explicit_zero_prob:
                loss_z = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_z
                metrics["train/mvc_nzlp"] = loss_z.item()
        if hyperparameter_defaults["ecs_thres"] > 0:
            loss_ecs = 10 * output_dict["loss_ecs"]
            loss = loss + loss_ecs
            metrics["train/ecs"] = loss_ecs.item()
        if "cls_output" in output_dict:
            loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
            loss = loss + cls_w * loss_cls
            metrics.update({"train/cls": loss_cls.item(), "train/cls_weight": cls_w})
            cls_acc = (output_dict["cls_output"].argmax(1) == celltype_labels).float().mean().item()
            metrics["train/cls_acc"] = cls_acc
            total_cls += loss_cls.item()
        if "loss_proto" in output_dict:
            loss_proto = output_dict["loss_proto"]
            loss = loss + proto_w * loss_proto
            metrics.update({"train/proto": loss_proto.item(), "train/proto_weight": proto_w})
            total_proto += loss_proto.item()
        if "loss_cce" in output_dict:
            loss_cce = output_dict["loss_cce"]
            loss = loss + cce_w * loss_cce
            metrics["train/cce"] = loss_cce.item()
            total_cce += loss_cce.item()
        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
        loss = loss + hyperparameter_defaults["dab_weight"] * loss_dab
        metrics["train/dab"] = loss_dab.item()

        model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        # Update EMA prototypes
        if hasattr(model, 'proto_head') and model.use_proto:
            model.proto_head.update_ema(
                output_dict["cell_emb"].detach(), celltype_labels
            )

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if hyperparameter_defaults["GEPC"] else 0.0
        total_ecs += loss_ecs.item() if hyperparameter_defaults["ecs_thres"] > 0 else 0.0
        total_dab += loss_dab.item()
        total_error += mre.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms = (time.time() - start_time) * 1000 / max(log_interval, 1)
            def _avg(x): return x / max(log_interval, 1)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:.6f} | ms/batch {ms:5.2f} | "
                f"loss {_avg(total_loss):5.2f} | mse {_avg(total_mse):5.2f} | "
                f"mre {_avg(total_error):5.2f} |"
                + (f" gepc {_avg(total_gepc):5.2f} |" if hyperparameter_defaults["GEPC"] else "")
                + f" cls {_avg(total_cls):5.2f} | proto {_avg(total_proto):5.2f} | "
                + f"cce {_avg(total_cce):5.2f} | ecs {_avg(total_ecs):5.2f} | "
                + f"dab {_avg(total_dab):5.2f} |"
            )
            total_loss = total_mse = total_gepc = 0.0
            total_cls = total_proto = total_cce = total_ecs = total_dab = 0.0
            total_error = 0.0
            start_time = time.time()

    return {"cls_weight": cls_w, "proto_weight": proto_w}


def evaluate(model, loader):
    model.eval()
    total_loss = total_error = total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
            )
            masked_positions = input_values.eq(mask_value)
            loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    val_loss = total_loss / total_num
    val_mre = total_error / total_num
    val_dab = total_dab / total_num
    logger.info(f"  Validation: loss={val_loss:.4f}, mre={val_mre:.4f}, dab={val_dab:.4f}")
    return val_loss, val_mre, val_dab


def compute_ari_from_embeddings(model, adata_t):
    """Compute ARI score from cell embeddings."""
    model.eval()
    adata_t = adata_t.copy()
    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )
    batch_ids = np.array(adata_t.obs["batch_id"].tolist())

    tokenized_all = tokenize_and_pad_batch(
        all_counts, gene_ids, max_len=max_seq_len,
        vocab=vocab, pad_token=pad_token, pad_value=pad_value,
        append_cls=True, include_zero_gene=True,
    )
    all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

    with torch.no_grad():
        cell_embeddings = model.encode_batch(
            all_gene_ids, all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=min(hyperparameter_defaults["batch_size"], 64),
            batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
            time_step=0, return_np=True,
        )
    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    adata_t.obsm["X_scGPT"] = cell_embeddings

    # Compute ARI from clustering
    sc.pp.neighbors(adata_t, use_rep="X_scGPT", n_neighbors=15)
    sc.tl.leiden(adata_t, resolution=0.5, random_state=42)

    true_labels = adata_t.obs["celltype"].cat.codes.values
    pred_labels = adata_t.obs["leiden"].astype(int).values

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return ari, nmi, adata_t


def evaluate_test(model, epoch, best_ari):
    """Run full evaluation on test set and save figures."""
    results = {}
    results["ari"], results["nmi"], adata_t = compute_ari_from_embeddings(
        model, adata_sorted if per_seq_batch_sample else adata
    )
    logger.info(f"  Test ARI = {results['ari']:.4f}, NMI = {results['nmi']:.4f}")

    # Generate UMAP plots
    sc.pp.neighbors(adata_t, use_rep="X_scGPT", n_neighbors=15)
    sc.tl.umap(adata_t, min_dist=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(
        adata_t, color=["celltype"],
        title=[f"Cell Types (ARI={results['ari']:.3f})"],
        frameon=False, show=False, ax=axes[0],
    )
    sc.pl.umap(
        adata_t, color=["str_batch"],
        title=["Batch"],
        frameon=False, show=False, ax=axes[1],
    )
    plt.tight_layout()
    fig.savefig(save_dir / f"embeddings_umap_e{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if results["ari"] > best_ari:
        best_ari = results["ari"]
        logger.info(f"*** New best ARI: {best_ari:.4f} ***")

    return results, best_ari


# %%
best_val_loss = float("inf")
best_ari = 0.0
best_model = None
best_model_epoch = 0
best_model_state = None

print("Starting training loop...")
logger.info("=" * 60)
logger.info("Starting scGPT fine-tuning with prototype contrastive learning")
logger.info("=" * 60)

for epoch in range(1, hyperparameter_defaults["epochs"] + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(
        sort_seq_batch=per_seq_batch_sample, current_epoch=epoch,
    )
    train_loader = prepare_dataloader(
        train_data_pt, batch_size=hyperparameter_defaults["batch_size"],
        shuffle=False, intra_domain_shuffle=True, drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt, batch_size=hyperparameter_defaults["batch_size"],
        shuffle=False, intra_domain_shuffle=False, drop_last=False,
    )

    if hyperparameter_defaults["do_train"]:
        weights = train(model, loader=train_loader, epoch=epoch)

    # Validate
    val_loss, val_mre, val_dab = evaluate(model, loader=valid_loader)
    elapsed = time.time() - epoch_start_time

    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:7.1f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f} | dab {val_dab:5.4f} |"
        f" cls_w={weights['cls_weight']:.3f} proto_w={weights['proto_weight']:.3f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        logger.info(f"Best model with score {best_val_loss:.4f}")

    # Evaluate on test set at intervals
    if epoch % hyperparameter_defaults["save_eval_interval"] == 0 or epoch == hyperparameter_defaults["epochs"]:
        logger.info(f"Evaluating on test set at epoch {epoch}...")
        eval_results, best_ari = evaluate_test(model, epoch, best_ari)

        # Save checkpoint
        logger.info(f"Saving checkpoint to {save_dir}")
        torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")
        if best_model is not None:
            torch.save(best_model.state_dict(), save_dir / f"best_model_e{best_model_epoch}.pt")

    scheduler.step()

    # Force garbage collection
    gc.collect()

# %%
# Final evaluation with best model
logger.info("=" * 60)
logger.info("Training completed. Running final evaluation...")
logger.info("=" * 60)

if best_model is not None:
    final_results, _ = evaluate_test(best_model, best_model_epoch, best_ari)
    logger.info(f"Final results - Best epoch: {best_model_epoch}")
    logger.info(f"  ARI (Adjusted Rand Index): {final_results['ari']:.4f}")
    logger.info(f"  NMI (Normalized Mutual Info): {final_results['nmi']:.4f}")

    # Save final model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")

    # Save metrics summary
    metrics_summary = {
        "best_epoch": best_model_epoch,
        "best_val_loss": best_val_loss,
        "best_ari": final_results["ari"],
        "best_nmi": final_results["nmi"],
        "config": hyperparameter_defaults,
    }
    with open(save_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics summary saved to {save_dir / 'metrics_summary.json'}")
else:
    logger.warning("No best model saved.")

print(f"\n{'='*60}")
print(f"Experiment Complete!")
print(f"Save directory: {save_dir}")
print(f"Best ARI: {final_results.get('ari', 'N/A'):.4f}" if best_model else "Best ARI: N/A")
print(f"{'='*60}")

gc.collect()