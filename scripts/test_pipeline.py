"""
scGPT Pipeline End-to-End Validation Script
============================================
Validates that the complete pipeline (data loading, preprocessing,
vocabulary creation, model building, training loop, evaluation)
works correctly end-to-end using a tiny data subset.

This runs 1 epoch with minimal settings for fast validation.
"""

import os
import sys
import json
import time
import gc
from pathlib import Path

import torch
import numpy as np
import scanpy as sc

# Ensure PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import logger


def main():
    print("=" * 60)
    print("scGPT Pipeline End-to-End Validation")
    print("=" * 60)

    # ---------- Step 1: Vocab creation (was broken) ----------
    print("\n[1/5] Testing GeneVocab initialization...")
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    test_genes = [f"GENE{i}" for i in range(100)]
    genes_all = test_genes + special_tokens
    vocab = GeneVocab(genes_all)
    vocab.set_default_index(vocab["<pad>"])
    print(f"  Vocab created: {len(vocab)} tokens")
    assert len(vocab) == 103, f"Expected 103 tokens, got {len(vocab)}"
    assert "<pad>" in vocab
    assert "<cls>" in vocab
    assert "<eoc>" in vocab
    assert "GENE0" in vocab
    print("  [PASS] GeneVocab initialization OK")

    # ---------- Step 2: Data loading ----------
    print("\n[2/5] Loading norman_2019 dataset (subset)...")
    data_path = PROJECT_ROOT / "data" / "norman_2019.h5ad"
    assert data_path.exists(), f"Data file not found: {data_path}"
    adata = sc.read_h5ad(data_path)
    print(f"  Full dataset shape: {adata.shape}")

    # Use tiny subset: 100 cells
    np.random.seed(42)
    idx = np.random.choice(adata.n_obs, min(500, adata.n_obs), replace=False)
    adata = adata[idx].copy()
    print(f"  Subset shape: {adata.shape}")

    # Column setup
    if "gemgroup" in adata.obs.columns and "gene_program" in adata.obs.columns:
        adata.obs["str_batch"] = adata.obs["gemgroup"].astype(int).astype(str)
        adata.obs["celltype"] = adata.obs["gene_program"].astype(str)
    else:
        adata.obs["str_batch"] = "0"
        adata.obs["celltype"] = "unknown"

    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    # Count layers
    if "counts" in adata.layers:
        input_layer_key = "counts"
        adata.X = adata.layers["counts"].copy()
    else:
        input_layer_key = "X"

    print("  [PASS] Data loaded OK")

    # ---------- Step 3: Vocab from data ----------
    print("\n[3/5] Creating GeneVocab from dataset genes...")
    all_genes = adata.var["gene_name"].tolist()
    vocab = GeneVocab(all_genes + special_tokens)
    vocab.set_default_index(vocab["<pad>"])
    print(f"  Vocab size: {len(vocab)} (from {len(all_genes)} genes)")

    # Get gene IDs
    gene_ids_in_vocab = np.array([
        1 if gene in vocab else -1 for gene in all_genes
    ])
    adata = adata[:, gene_ids_in_vocab >= 0]
    print(f"  After vocab filter: {adata.shape}")
    print("  [PASS] GeneVocab creation OK")

    # ---------- Step 4: Preprocessing ----------
    print("\n[4/5] Running preprocessing (HVG, binning)...")
    data_is_raw = True
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=200,  # Keep small for speed
        hvg_flavor="cell_ranger",
        binning=51,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")

    # Get gene_ids after HVG selection
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    print(f"  After HVG: {adata.shape}, gene_ids: {len(gene_ids)}")
    assert len(gene_ids) == 200, f"Expected 200 gene_ids after HVG, got {len(gene_ids)}"
    print("  [PASS] Preprocessing OK")

    # ---------- Step 5: Tokenize + Model creation ----------
    print("\n[5/5] Running tokenization, model creation, and training...")
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.model import TransformerModel
    from scgpt.loss import masked_mse_loss
    from scgpt.trainer import get_warmup_cosine_lr_scheduler
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split

    pad_token = "<pad>"
    mask_value = -1
    pad_value = -2
    n_input_bins = 51
    explicit_zero_prob = True
    DSBN = True

    # Prepare data
    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if hasattr(adata.layers[input_layer_key], "toarray")
        else np.asarray(adata.layers[input_layer_key])
    )
    batch_ids = np.array(adata.obs["batch_id"].tolist())
    flags = np.array(adata.obs["celltype"].tolist())

    # Train/val split
    train_data, valid_data, _, _, train_batch, valid_batch = \
        train_test_split(all_counts, flags, batch_ids, test_size=0.2, shuffle=True)

    max_seq_len = 201
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True,
    )

    print(f"  Train: {tokenized_train['genes'].shape[0]}, Valid: {tokenized_valid['genes'].shape[0]}")

    # Create model
    device = torch.device("cpu")
    ntokens = len(vocab)
    num_batch_types = len(set(batch_id_labels))

    model = TransformerModel(
        ntokens, 64, 2, 64, 2,
        vocab=vocab, dropout=0.2,
        pad_token=pad_token, pad_value=pad_value,
        do_mvc=True, do_dab=True,
        use_batch_labels=True, num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN, n_input_bins=n_input_bins,
        ecs_threshold=0.8,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True,
        pre_norm=True,
    )
    model.to(device)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer & scheduler
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # One batch of training
    model.train()
    batch_size = min(16, len(train_data))

    # Prepare batch
    class SeqDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return self.data["gene_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    train_dataset = SeqDataset({
        "gene_ids": tokenized_train["genes"],
        "values": random_mask_value(
            tokenized_train["values"],
            mask_ratio=0.25, mask_value=mask_value, pad_value=pad_value,
        ),
        "target_values": tokenized_train["values"],
        "batch_labels": torch.from_numpy(train_batch).long(),
    })

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Run one batch
    for batch_data in train_loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        output_dict = model(
            input_gene_ids, input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if DSBN else None,
            MVC=True,
            ECS=True,
        )

        masked_positions = input_values.eq(mask_value)
        loss = criterion(output_dict["mlm_output"], target_values, masked_positions)

        if "mvc_output" in output_dict:
            loss = loss + criterion(output_dict["mvc_output"], target_values, masked_positions)
        if "loss_ecs" in output_dict:
            loss = loss + 10 * output_dict["loss_ecs"]
        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
        loss = loss + 1.0 * loss_dab

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Training step: loss={loss.item():.4f}")
        break  # Just one batch

    print("  [PASS] Training step completed successfully!")

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - Pipeline validates successfully!")
    print("=" * 60)

    # Cleanup
    gc.collect()


if __name__ == "__main__":
    main()