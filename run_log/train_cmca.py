#!/usr/bin/env python3
"""
scGPT Multi-Omic Training with Cross-Modal Contrastive Alignment (CMCA)

Optimized training pipeline for single-cell genomics data with:
  - Cross-Modal Contrastive Alignment (CMCA) Loss
  - Masked Language Modeling (GEP/MLM)
  - Masked Value Prediction (GEPC/MVC)
  - Elastic Cell Similarity (ECS)
  - Domain Adversarial Training (DAB)
  - Validation, Testing, and scIB metrics evaluation
  - Checkpoint resume
  - Gradient accumulation for CPU efficiency
"""
import sys, os, copy, gc, json, time, warnings, math, traceback
from types import SimpleNamespace
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Triton compatibility workaround for CPU-only environments
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
try:
    import triton as _triton
except ImportError:
    _triton = None

if _triton is not None and not hasattr(_triton, 'language'):
    _stub_modules = [
        'triton.language', 'triton.backends', 'triton.backends.compiler',
        'triton.compiler', 'triton.compiler.compiler',
        'triton.runtime', 'triton.runtime.driver', 'triton.ops',
        'triton.language.extra', 'triton.language.standard',
    ]
    for _mod_path in _stub_modules:
        if _mod_path not in _sys.modules:
            _parts = _mod_path.split('.')
            _mod = _types.ModuleType(_mod_path)
            _mod.__path__ = []
            _mod.__package__ = _mod_path
            _sys.modules[_mod_path] = _mod
            _parent_path = '.'.join(_parts[:-1])
            if _parent_path and _parent_path in _sys.modules:
                setattr(_sys.modules[_parent_path], _parts[-1], _mod)
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLE = 500
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over N steps on CPU

_config_dict = dict(
    seed=42,
    mask_ratio=0.4,
    epochs=5,                       # More epochs for better convergence
    n_bins=51,
    GEPC=True,
    ecs_thres=0.8,
    dab_weight=0.5,                 # Reduced DAB weight to focus on GEP
    lr=1.5e-4,                      # Slightly lower LR for stable training
    batch_size=8,
    layer_size=128,                  # Increased for better representation
    nlayers=4,                       # Deeper model for better accuracy
    nhead=8,                         # More attention heads
    dropout=0.15,                    # Slightly lower dropout
    schedule_ratio=0.9,
    save_eval_interval=1,
    log_interval=5,                  # More frequent logging
    fast_transformer=True,
    pre_norm=True,                   # Pre-layer norm for training stability
    amp=True,                        # Will be auto-disabled on CPU
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
    cmca_weight=0.2,                 # Slightly higher CMCA weight
    cmca_temp=0.5,
    warmup_epochs=1,                  # Warmup for LR stability
    weight_decay=0.02,                # Stronger regularization
    early_stop_patience=3,            # Early stopping patience
)
config = SimpleNamespace(**_config_dict)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cpu = device.type == "cpu"

# Disable AMP on CPU (autocast is CUDA-specific; on CPU it adds overhead)
if is_cpu:
    config.amp = False

logger = scg.logger
save_dir = Path(OUTPUT_DIR)
save_dir.mkdir(parents=True, exist_ok=True)
scg.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Output directory: {save_dir}")
logger.info(f"Device: {device}")
logger.info(f"AMP enabled: {config.amp}")
logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")

set_seed(config.seed)

# ============================================================================
# Data loading
# ============================================================================
import scanpy as sc
logger.info("Loading norman_2019.h5ad...")
adata = sc.read_h5ad(os.path.join(PROJECT_ROOT, "data", "norman_2019.h5ad"))

# Subsample for fast testing
if adata.n_obs > N_SAMPLE:
    np.random.seed(config.seed)
    idx = np.random.choice(adata.n_obs, N_SAMPLE, replace=False)
    adata = adata[idx].copy()
    logger.info(f"Subsampled to {adata.n_obs} cells for fast testing")

logger.info(f"Data shape: {adata.shape}")
logger.info(f"Obs columns: {list(adata.obs.columns)}")

# Setup cell type and batch labels
if "cell_type" in adata.obs.columns:
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
elif "celltype" not in adata.obs.columns:
    adata.obs["celltype"] = "unknown"

if "perturbation" in adata.obs.columns:
    adata.obs["str_batch"] = adata.obs["perturbation"].astype(str)
elif "batch" in adata.obs.columns:
    adata.obs["str_batch"] = adata.obs["batch"].astype(str)
else:
    adata.obs["str_batch"] = "0"

batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Create pseudo-modalities: split genes into two groups
# First half = RNA (mod 0), Second half = Protein (mod 1)
n_genes_total = adata.n_vars
mod_type_array = np.zeros(n_genes_total, dtype=np.int64)
if n_genes_total > 1:
    split_idx = n_genes_total // 2
    mod_type_array[split_idx:] = 1

n_hvg = min(200, n_genes_total)
max_seq_len = n_hvg + 1  # +1 for CLS token

# Special tokens
special_tokens = [config.pad_token, "<cls>", "<eoc>"]

# Build vocabulary
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

# Train/valid/test split (70/15/15)
train_data, test_data, train_cl, test_cl, train_bl, test_bl = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.15, shuffle=True, random_state=config.seed
)
train_data, valid_data, train_cl, valid_cl, train_bl, valid_bl = train_test_split(
    train_data, train_cl, train_bl, test_size=0.1765, shuffle=True, random_state=config.seed
)
logger.info(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

gene_ids = np.array(vocab(genes), dtype=int)

# Modality types for HVG-selected genes
mod_type_hvg = mod_type_array[:n_hvg] if n_hvg <= len(mod_type_array) else np.pad(
    mod_type_array, (0, n_hvg - len(mod_type_array)), 'constant'
)

# Tokenize
logger.info("Tokenizing datasets...")
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
tokenized_test = tokenize_and_pad_batch(
    test_data, gene_ids[:n_hvg], max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value,
    append_cls=True, include_zero_gene=True,
    mod_type=mod_type_hvg, vocab_mod=vocab,
)

logger.info(f"Train samples: {len(tokenized_train['genes'])}, "
            f"Valid samples: {len(tokenized_valid['genes'])}, "
            f"Test samples: {len(tokenized_test['genes'])}")

# ============================================================================
# Build model
# ============================================================================
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

# ============================================================================
# Data loaders
# ============================================================================
from scgpt.trainer import prepare_data as sc_prepare_data

def _make_loader(data_pt, batch_size, shuffle=False, intra_shuffle=False, drop_last=False):
    """Create a DataLoader with SubsetsBatchSampler for batch-aware sampling."""
    class SeqDataset(Dataset):
        def __init__(self, d):
            self.data = d
        def __len__(self):
            return self.data["gene_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    dataset = SeqDataset(data_pt)
    bl = data_pt["batch_labels"].numpy() if torch.is_tensor(data_pt["batch_labels"]) else np.array(data_pt["batch_labels"])
    subsets = []
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
        pin_memory=False if is_cpu else True,
    )

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr,
    weight_decay=config.weight_decay, eps=1e-4,
    betas=(0.9, 0.98),                 # AdamW betas
)

# Warmup + Cosine Annealing scheduler for better convergence
def _get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """Linear warmup followed by cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(eta_min / config.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = _get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs=config.warmup_epochs,
    total_epochs=config.epochs,
)   # Use LambdaLR for warmup + cosine

# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------
resume_ckpt = os.environ.get("LOAD_MODEL", "")
start_epoch = 1
best_val_loss = float("inf")
best_model_state = None
best_epoch = 0
early_stop_counter = 0

if resume_ckpt and os.path.isfile(resume_ckpt):
    logger.info(f"Loading checkpoint: {resume_ckpt}")
    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    best_epoch = ckpt.get("best_epoch", 0)
    logger.info(f"Resumed from epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})")

# ============================================================================
# Helper functions (defined before training loop to avoid NameError)
# ============================================================================
def _forward_model(model, ig, iv, src_mask, bl, mt, config, vocab):
    """Forward pass through the model."""
    return model(
        ig, iv,
        src_key_padding_mask=src_mask,
        batch_labels=bl if config.DSBN else None,
        MVC=config.GEPC,
        ECS=config.ecs_thres > 0,
        CMCA=config.CMCA,
        mod_types=mt if config.use_mod else None,
    )


def _compute_loss(out, ig, iv, tv, bl, mt, config, vocab, criterion, criterion_dab, epoch):
    """Compute combined training loss with all components."""
    masked_pos = iv.eq(config.mask_value)
    loss = 0.0

    # GEP (MLM) loss
    loss_gep = criterion(out["mlm_output"], tv, masked_pos)
    loss += loss_gep
    out["gep_loss"] = loss_gep.item()

    if config.explicit_zero_prob and "mlm_zero_probs" in out:
        loss += criterion_neg_log_bernoulli(out["mlm_zero_probs"], tv, masked_pos)

    # GEPC (MVC) loss
    if config.GEPC:
        loss_gepc = criterion(out["mvc_output"], tv, masked_pos)
        loss += loss_gepc
        out["gepc_loss"] = loss_gepc.item()
        if config.explicit_zero_prob and "mvc_zero_probs" in out:
            loss += criterion_neg_log_bernoulli(out["mvc_zero_probs"], tv, masked_pos)

    # ECS loss
    if config.ecs_thres > 0:
        loss_ecs = 10 * out["loss_ecs"]
        loss += loss_ecs
        out["ecs_loss"] = loss_ecs.item()

    # DAB loss
    loss_dab = criterion_dab(out["dab_output"], bl)
    loss += config.dab_weight * loss_dab
    out["dab_loss"] = loss_dab.item()

    # CMCA loss
    if config.CMCA and "loss_cmca" in out:
        cmca_loss_val = out["loss_cmca"]
        loss += config.cmca_weight * cmca_loss_val
        out["cmca_loss"] = cmca_loss_val.item()
    else:
        out["cmca_loss"] = 0.0

    return loss


# ============================================================================
# Training loop
# ============================================================================
train_start = time.time()

for epoch in range(start_epoch, config.epochs + 1):
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
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(train_loader):
        ig = batch_data["gene_ids"].to(device)
        iv = batch_data["values"].to(device)
        tv = batch_data["target_values"].to(device)
        bl = batch_data["batch_labels"].to(device)
        mt = batch_data.get("mod_types", None)
        if mt is not None:
            mt = mt.to(device)

        src_mask = ig.eq(vocab[config.pad_token])

        # Forward pass without AMP on CPU
        if config.amp:
            with torch.cuda.amp.autocast():
                out = _forward_model(model, ig, iv, src_mask, bl, mt, config, vocab)
                loss = _compute_loss(out, ig, iv, tv, bl, mt, config, vocab,
                                     criterion, criterion_dab, epoch)
        else:
            out = _forward_model(model, ig, iv, src_mask, bl, mt, config, vocab)
            loss = _compute_loss(out, ig, iv, tv, bl, mt, config, vocab,
                                 criterion, criterion_dab, epoch)

        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # Accumulate metrics
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_gep += out.get("gep_loss", 0)
        if config.GEPC:
            total_gepc += out.get("gepc_loss", 0)
        if config.ecs_thres > 0:
            total_ecs += out.get("ecs_loss", 0)
        if config.DAR:
            total_dab += out.get("dab_loss", 0)
        if config.CMCA:
            total_cmca += out.get("cmca_loss", 0)

        # Gradient accumulation step
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()

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
    val_gep = val_dab = 0.0
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
            out = model(ig, iv, src_key_padding_mask=src_mask,
                        batch_labels=bl if config.DSBN else None,
                        MVC=False, ECS=False, CMCA=False,
                        mod_types=mt if config.use_mod else None)
            masked_pos = iv.eq(config.mask_value)
            loss_gep = criterion(out["mlm_output"], tv, masked_pos)
            loss_dab = criterion_dab(out["dab_output"], bl)
            loss = loss_gep + config.dab_weight * loss_dab
            val_loss += loss.item() * len(ig)
            val_gep += loss_gep.item() * len(ig)
            val_dab += loss_dab.item() * len(ig)
            val_n += len(ig)

    val_loss /= val_n
    val_gep /= val_n
    val_dab /= val_n
    logger.info(f"| end ep {epoch:3d} | {time.time()-epoch_start:5.2f}s | "
                f"val_loss {val_loss:.4f} (gep {val_gep:.4f}, dab {val_dab:.4f}) | "
                f"lr {scheduler.get_last_lr()[0]:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        logger.info(f"  -> Best model (ep {epoch}, val_loss {val_loss:.4f})")

    # Save checkpoint
    ckpt_path = save_dir / f"checkpoint_ep{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "config": _config_dict,
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    scheduler.step()

    # --- Early Stopping ---
    if val_loss >= best_val_loss:
        early_stop_counter += 1
        logger.info(f"  Early stopping counter: {early_stop_counter}/{config.early_stop_patience}")
        if early_stop_counter >= config.early_stop_patience:
            logger.info(f"  >> Early stopping triggered after epoch {epoch} (val_loss did not improve for {config.early_stop_patience} epochs)")
            break
    else:
        early_stop_counter = 0

total_time = time.time() - train_start
logger.info(f"Training completed: {total_time:.1f}s ({total_time/60:.1f} min)")

# Save best model
if best_model_state is not None:
    torch.save(best_model_state, save_dir / "best_model.pt")
    logger.info(f"Best model (ep {best_epoch}, val_loss {best_val_loss:.4f}) saved.")

# ============================================================================
# Test Phase
# ============================================================================
logger.info("=" * 60)
logger.info("Starting Test Phase...")

# Reload best model for testing
test_model = MultiOmicTransformerModel(
    ntoken=ntokens, d_model=config.layer_size,
    nhead=config.nhead, d_hid=config.layer_size,
    nlayers=config.nlayers, vocab=vocab,
    dropout=config.dropout, pad_token=config.pad_token,
    pad_value=config.pad_value, do_mvc=config.GEPC,
    do_dab=True, use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    use_mod=True, ntokens_mod=ntokens, vocab_mod=vocab,
    do_cmca=False,
)

if best_model_state is not None:
    test_model.load_state_dict(best_model_state)
else:
    test_model.load_state_dict(model.state_dict())
test_model.to(device).eval()
logger.info("Loaded best model for testing.")

# Prepare test data
test_pt, _ = sc_prepare_data(
    tokenized_test, tokenized_test,
    test_bl, test_bl, config, 0,
    sort_seq_batch=True,
)
test_loader = _make_loader(test_pt, config.batch_size,
                            shuffle=False, intra_shuffle=False)

# Test: compute MLM loss and DAB accuracy
test_loss = test_n = 0
test_gep_total = test_dab_total = 0.0
test_dab_correct = 0
with torch.no_grad():
    for batch_data in test_loader:
        ig = batch_data["gene_ids"].to(device)
        iv = batch_data["values"].to(device)
        tv = batch_data["target_values"].to(device)
        bl = batch_data["batch_labels"].to(device)
        mt = batch_data.get("mod_types", None)
        if mt is not None:
            mt = mt.to(device)

        src_mask = ig.eq(vocab[config.pad_token])
        out = test_model(ig, iv, src_key_padding_mask=src_mask,
                         batch_labels=bl if config.DSBN else None,
                         MVC=False, ECS=False, CMCA=False,
                         mod_types=mt if config.use_mod else None)
        masked_pos = iv.eq(config.mask_value)
        loss_gep = criterion(out["mlm_output"], tv, masked_pos)
        loss_dab = criterion_dab(out["dab_output"], bl)
        loss = loss_gep + config.dab_weight * loss_dab

        test_loss += loss.item() * len(ig)
        test_gep_total += loss_gep.item() * len(ig)
        test_dab_total += loss_dab.item() * len(ig)
        test_n += len(ig)

        # DAB accuracy
        dab_pred = out["dab_output"].argmax(dim=1)
        test_dab_correct += (dab_pred == bl).sum().item()

test_loss /= test_n
test_gep_total /= test_n
test_dab_total /= test_n
test_dab_acc = test_dab_correct / test_n

logger.info(f"Test Results: loss={test_loss:.4f}, gep={test_gep_total:.4f}, "
            f"dab_loss={test_dab_total:.4f}, dab_acc={test_dab_acc:.4f}")

# Generate cell embeddings and compute scIB metrics
logger.info("Generating cell embeddings for scIB evaluation...")
try:
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
        mod_type=mod_type_hvg, vocab_mod=vocab,
    )
    all_gids, all_vals = tokenized_all["genes"], tokenized_all["values"]
    src_pad = all_gids.eq(vocab[config.pad_token])

    with torch.no_grad():
        cell_embs = test_model.encode_batch(
            all_gids, all_vals.float(), src_pad,
            batch_size=config.batch_size,
            batch_labels=torch.from_numpy(batch_ids_eval).long(),
            time_step=0, return_np=True,
        )
    cell_embs = cell_embs / np.linalg.norm(cell_embs, axis=1, keepdims=True)
    adata_sorted.obsm["X_scGPT"] = cell_embs

    results = eval_scib_metrics(adata_sorted)
    for k, v in results.items():
        logger.info(f"scIB {k}: {v:.4f}" if isinstance(v, float) else f"scIB {k}: {v}")
except Exception as e:
    logger.error(f"scIB evaluation failed: {e}")
    traceback.print_exc()

logger.info("=" * 60)
logger.info("Pipeline finished successfully!")
gc.collect()
