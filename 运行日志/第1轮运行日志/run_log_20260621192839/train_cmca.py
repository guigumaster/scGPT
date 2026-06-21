#!/usr/bin/env python3
"""
scGPT Multi-Omic Training with Cross-Modal Contrastive Alignment Loss (CMCA-Loss)

This script provides a complete training, validation, and testing pipeline for
the scGPT model with CMCA-Loss for cross-modal alignment.

Data: Uses the local norman_2019.h5ad perturbation dataset.
"""
import sys, os, copy, gc, json, time, warnings, math, traceback
from types import SimpleNamespace
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Triton compatibility workaround
# The installed triton package is an empty stub missing submodules like
# ``triton.language`` and ``triton.backends`` that PyTorch 2.x expects.
# Since we run on CPU, we patch triton with minimal stubs registered in
# sys.modules so that ``import triton.language`` and friends work correctly.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
try:
    import triton as _triton
except ImportError:
    _triton = None

if _triton is not None and not hasattr(_triton, 'language'):
    # Build a tree of stub sub-packages that torch._dynamo / torch._inductor
    # expect to exist.  Each sub-package must be registered in sys.modules
    # so that regular ``import triton.xxx.yyy`` statements resolve.
    _stub_modules = [
        'triton.language',
        'triton.backends',
        'triton.backends.compiler',
        'triton.compiler',
        'triton.compiler.compiler',
        'triton.runtime',
        'triton.runtime.driver',
        'triton.ops',
        'triton.language.extra',
        'triton.language.standard',
    ]
    for _mod_path in _stub_modules:
        if _mod_path not in _sys.modules:
            _parts = _mod_path.split('.')
            _parent_path = '.'.join(_parts[:-1])
            _mod = _types.ModuleType(_mod_path)
            _mod.__path__ = []       # mark as a package so sub-imports work
            _mod.__package__ = _mod_path
            _sys.modules[_mod_path] = _mod
            # Wire into parent namespace
            if _parent_path and _parent_path in _sys.modules:
                setattr(_sys.modules[_parent_path], _parts[-1], _mod)
    # Add the ``dtype`` sentinel that torch._dynamo.utils expects
    _triton.language.dtype = type('dtype', (), {'__repr__': lambda _s: 'dtype'})

import torch
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "save"))

sys.path.insert(0, PROJECT_ROOT)

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import MultiOmicTransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics

# Use a small subset for fast CPU testing
N_SAMPLE = 500

# ---------------------------------------------------------------------------
# Configuration (use SimpleNamespace so both dict-style and attribute access work)
# ---------------------------------------------------------------------------
_config_dict = dict(
    seed=42,
    mask_ratio=0.4,
    epochs=2,
    n_bins=51,
    GEPC=True,
    ecs_thres=0.8,
    dab_weight=1.0,
    lr=2e-4,
    batch_size=8,
    layer_size=64,
    nlayers=2,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=1,
    log_interval=10,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    task="multiomic",
    GEP=True,
    CLS=False,
    ESC=True,
    DAR=True,
    DSBN=True,
    use_batch_labels=True,
    use_mod=True,
    explicit_zero_prob=True,
    pad_token="<pad>",
    mask_value=-1,
    pad_value=-2,
    include_zero_gene=True,
    n_input_bins=51,
    max_seq_len=1201,
    input_layer_key="X_binned",
    CMCA=True,
    cmca_weight=0.1,
    cmca_temp=0.5,
)
config = SimpleNamespace(**_config_dict)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = scg.logger
save_dir = Path(OUTPUT_DIR)
save_dir.mkdir(parents=True, exist_ok=True)
scg.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Output directory: {save_dir}")
logger.info(f"Device: {device}")

set_seed(config.seed)

# ============================================================================
# Data loading - using local norman_2019.h5ad
# ============================================================================
import scanpy as sc
logger.info("Loading norman_2019.h5ad...")
adata = sc.read_h5ad(os.path.join(PROJECT_ROOT, "data", "norman_2019.h5ad"))
# Subsample for fast CPU testing
if adata.n_obs > N_SAMPLE:
    np.random.seed(config.seed)
    idx = np.random.choice(adata.n_obs, N_SAMPLE, replace=False)
    adata = adata[idx].copy()
    logger.info(f"Subsampled to {adata.n_obs} cells for fast testing")

# Prepare data for multi-omic training
# We'll split genes into two pseudo-modalities for CMCA demonstration
logger.info(f"Data shape: {adata.shape}")
logger.info(f"Obs columns: {list(adata.obs.columns)}")

# Use existing cell types and create batch labels
if "cell_type" in adata.obs.columns:
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
elif "celltype" not in adata.obs.columns:
    adata.obs["celltype"] = "unknown"

# Create batch labels (use perturbation if available, else create single batch)
if "perturbation" in adata.obs.columns:
    adata.obs["str_batch"] = adata.obs["perturbation"].astype(str)
elif "batch" in adata.obs.columns:
    adata.obs["str_batch"] = adata.obs["batch"].astype(str)
else:
    adata.obs["str_batch"] = "0"

batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Create pseudo-modalities: split genes roughly in half
# First half = "RNA" (mod 0), Second half = "Protein" (mod 1)
n_genes_total = adata.n_vars
mod_type_array = np.zeros(n_genes_total, dtype=np.int64)
mod_type_array[n_genes_total // 2:] = 1  # Second half = Protein

n_hvg = min(200, n_genes_total)
max_seq_len = n_hvg + 1  # +1 for CLS token

# Special tokens
special_tokens = [config.pad_token, "<cls>", "<eoc>"]

# Build vocabulary using scgpt's GeneVocab
genes = adata.var["gene_name"].tolist()
from scgpt.tokenizer.gene_tokenizer import GeneVocab
vocab = GeneVocab(
    gene_list_or_vocab=genes + special_tokens,
    specials=special_tokens,
    special_first=True,
)
vocab.set_default_index(vocab[config.pad_token])

# Preprocess
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=3,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=False,
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,
    hvg_flavor="seurat_v3",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch")

# Sort by batch
adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

input_key = config.input_layer_key
all_counts = (
    adata.layers[input_key].toarray()
    if issparse(adata.layers[input_key])
    else adata.layers[input_key]
)
celltypes_labels = np.array(adata.obs["celltype"].tolist())
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(np.unique(batch_ids))

# Train/valid split
(train_data, valid_data, train_cl, valid_cl,
 train_bl, valid_bl) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

gene_ids = np.array(vocab(genes), dtype=int)

# Subset mod_type_array to match HVG selection
mod_type_hvg = mod_type_array.copy()
# After HVG selection, we need to map back. Let's get the HVG indices
hvg_idx = adata.var["highly_variable"].values if "highly_variable" in adata.var.columns else np.arange(n_hvg)
# Use the first n_hvg genes' mod types
mod_type_hvg = mod_type_array[:n_hvg] if n_hvg <= len(mod_type_array) else np.pad(mod_type_array, (0, n_hvg - len(mod_type_array)), 'constant')

# Tokenize with mod types for CMCA
tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids[:n_hvg], max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value,
    append_cls=True, include_zero_gene=True,
    mod_type=mod_type_hvg, vocab_mod=vocab,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids[:n_hvg], max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value,
    append_cls=True, include_zero_gene=True,
    mod_type=mod_type_hvg, vocab_mod=vocab,
)

logger.info(f"Train samples: {len(tokenized_train['genes'])}, Valid samples: {len(tokenized_valid['genes'])}")
logger.info(f"mod_types in train: {tokenized_train.get('mod_types', None)}")

# ---------------------------------------------------------------------------
# Build model with CMCA
# ---------------------------------------------------------------------------
ntokens = len(vocab)
logger.info(f"Vocabulary size: {ntokens}")

model = MultiOmicTransformerModel(
    ntoken=ntokens,
    d_model=config.layer_size,
    nhead=config.nhead,
    d_hid=config.layer_size,
    nlayers=config.nlayers,
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

model.to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Model parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
from scgpt.trainer import prepare_data as sc_prepare_data

def _make_loader(data_pt, batch_size, shuffle=False, intra_shuffle=False, drop_last=False):
    class SeqDataset(Dataset):
        def __init__(self, d):
            self.data = d
        def __len__(self):
            return self.data["gene_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    dataset = SeqDataset(data_pt)
    subsets = []
    bl = data_pt["batch_labels"].numpy() if torch.is_tensor(data_pt["batch_labels"]) else np.array(data_pt["batch_labels"])
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
        num_workers=0,
        pin_memory=True,
    )

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr,
    weight_decay=0.01, eps=1e-4 if config.amp else 1e-8,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs, eta_min=1e-6
)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# ============================================================================
# Training loop
# ============================================================================
best_val_loss = float("inf")
best_model_state = None
best_epoch = 0
train_start = time.time()

for epoch in range(1, config.epochs + 1):
    epoch_start = time.time()

    # Prepare data with masking
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
    total_loss = total_cmca = total_gep = total_gepc = total_dab = total_ecs = 0.0
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

            loss = 0.0
            loss_gep = criterion(out["mlm_output"], tv, masked_pos)
            loss += loss_gep; total_gep += loss_gep.item()

            if config.explicit_zero_prob and "mlm_zero_probs" in out:
                loss += criterion_neg_log_bernoulli(out["mlm_zero_probs"], tv, masked_pos)

            if config.GEPC:
                loss_gepc = criterion(out["mvc_output"], tv, masked_pos)
                loss += loss_gepc; total_gepc += loss_gepc.item()
                if config.explicit_zero_prob and "mvc_zero_probs" in out:
                    loss += criterion_neg_log_bernoulli(out["mvc_zero_probs"], tv, masked_pos)

            if config.ecs_thres > 0:
                loss_ecs = 10 * out["loss_ecs"]
                loss += loss_ecs; total_ecs += loss_ecs.item()

            loss_dab = criterion_dab(out["dab_output"], bl)
            loss += config.dab_weight * loss_dab; total_dab += loss_dab.item()

            cmca_loss_val = 0.0
            if config.CMCA and "loss_cmca" in out:
                cmca_loss_val = out["loss_cmca"]
                loss += config.cmca_weight * cmca_loss_val
                total_cmca += cmca_loss_val.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if batch_idx % config.log_interval == 0 and batch_idx > 0:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"| ep {epoch:3d} | {batch_idx:3d}/{n_batches:3d} | "
                f"lr {lr_now:.2e} | loss {total_loss/(batch_idx+1):5.2f} | "
                f"gep {total_gep/(batch_idx+1):5.2f} | "
                + (f"gepc {total_gepc/(batch_idx+1):5.2f} |" if config.GEPC else "")
                + (f"ecs {total_ecs/(batch_idx+1):5.2f} |" if config.ecs_thres > 0 else "")
                + (f"dab {total_dab/(batch_idx+1):5.2f} |" if config.DAR else "")
                + (f"cmca {total_cmca/(batch_idx+1):5.4f} |" if config.CMCA else "")
            )

    # --- Validation ---
    model.eval()
    val_loss = val_n = 0
    with torch.no_grad():
        for batch_data in valid_loader:
            ig = batch_data["gene_ids"].to(device)
            iv = batch_data["values"].to(device)
            tv = batch_data["target_values"].to(device)
            bl = batch_data["batch_labels"].to(device)
            mt = batch_data.get("mod_types", None)
            if mt is not None: mt = mt.to(device)

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
    logger.info(f"| end ep {epoch:3d} | {time.time()-epoch_start:5.2f}s | val_loss {val_loss:5.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        logger.info(f"  -> Best model (ep {epoch}, val_loss {val_loss:.4f})")

    # Periodic save
    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        if best_model_state is not None:
            ckpt_path = save_dir / f"model_ep{best_epoch}.pt"
            torch.save(best_model_state, ckpt_path)
            logger.info(f"Saved to {ckpt_path}")

    scheduler.step()

# --- Final save ---
total_time = time.time() - train_start
logger.info(f"Training: {total_time:.1f}s ({total_time/60:.1f} min)")
if best_model_state is not None:
    torch.save(best_model_state, save_dir / "best_model.pt")
    logger.info(f"Best model (ep {best_epoch}, val_loss {best_val_loss:.4f}) saved.")

# --- Evaluate ---
logger.info("Evaluating...")
try:
    eval_model = MultiOmicTransformerModel(
        ntoken=ntokens, d_model=config.layer_size,
        nhead=config.nhead, d_hid=config.layer_size,
        nlayers=config.nlayers, vocab=vocab,
        dropout=config.dropout, pad_token=config.pad_token,
        pad_value=config.pad_value, do_mvc=False, do_dab=False,
        use_batch_labels=False, domain_spec_batchnorm=False,
        n_input_bins=config.n_input_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=False, use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm, use_mod=False,
    )
    eval_model.load_state_dict(best_model_state)
    eval_model.to(device).eval()

    all_counts_eval = (
        adata_sorted.layers[input_key].toarray()
        if issparse(adata_sorted.layers[input_key])
        else adata_sorted.layers[input_key]
    )
    batch_ids_eval = np.array(adata_sorted.obs["batch_id"].tolist())

    tokenized_all = tokenize_and_pad_batch(
        all_counts_eval, gene_ids[:n_hvg],
        max_len=max_seq_len, vocab=vocab,
        pad_token=config.pad_token, pad_value=config.pad_value,
        append_cls=True, include_zero_gene=config.include_zero_gene,
    )
    all_gids, all_vals = tokenized_all["genes"], tokenized_all["values"]
    src_pad = all_gids.eq(vocab[config.pad_token])

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        cell_embs = eval_model.encode_batch(
            all_gids, all_vals.float(), src_pad,
            batch_size=config.batch_size,
            batch_labels=torch.from_numpy(batch_ids_eval).long(),
            time_step=0, return_np=True,
        )
    cell_embs = cell_embs / np.linalg.norm(cell_embs, axis=1, keepdims=True)
    adata_sorted.obsm["X_scGPT"] = cell_embs

    results = eval_scib_metrics(adata_sorted)
    for k, v in results.items():
        logger.info(f"Final {k}: {v:.4f}" if isinstance(v, float) else f"Final {k}: {v}")
except Exception as e:
    logger.error(f"Eval failed: {e}")
    traceback.print_exc()

logger.info("Done.")
gc.collect()