"""
Norman perturbation dataset (CRISPR single/double gene knockout scRNA-seq) loader.

This module provides utilities to download, preprocess, and create DataLoaders
for the Norman et al. 2019 perturbation dataset.

Reference:
    Norman, T. M. et al. (2019). Exploring genetic interaction manifolds
    constructed from rich single-cell phenotypes. Science, 365(6455), eaax4438.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.data_sampler import SubsetsBatchSampler
from scgpt import logger


def load_norman_h5ad(
    data_path: Union[str, Path] = "./tutorials/data/norman.h5ad",
) -> AnnData:
    """
    Load the pre-processed Norman CRISPR perturbation scRNA-seq dataset from h5ad.

    Args:
        data_path: Path to the norman.h5ad file.

    Returns:
        AnnData object with raw expression counts in .X
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Norman data not found at {data_path}. "
            "Please download the Norman dataset from "
            "https://dataverse.harvard.edu/api/access/datafile/6154020 "
            "and place it at the expected location."
        )
    logger.info(f"Loading Norman data from {data_path}")
    adata = sc.read_h5ad(data_path)
    return _annotate_norman(adata)


def _annotate_norman(adata: AnnData) -> AnnData:
    """Add standardized column names expected by the training pipeline."""
    # --- perturbation label ---
    if "perturbation" not in adata.obs:
        for candidate in ["perturbation", "gene", "target", "condition", "gene_target"]:
            if candidate in adata.obs:
                adata.obs["perturbation"] = adata.obs[candidate].astype(str)
                break
        else:
            adata.obs["perturbation"] = "ctrl"

    # --- cell type ---
    if "celltype" not in adata.obs:
        adata.obs["celltype"] = "K562"

    # --- str_batch / batch_id ---
    if "str_batch" not in adata.obs or "batch_id" not in adata.obs:
        if "batch" in adata.obs:
            adata.obs["str_batch"] = adata.obs["batch"].astype(str)
        else:
            adata.obs["str_batch"] = "0"
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels

    # --- gene name column ---
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var.index.tolist()

    logger.info(
        f"Norman data loaded: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs['str_batch'].nunique()} batches, "
        f"{adata.obs['perturbation'].nunique()} perturbations."
    )
    return adata


def preprocess_norman_adata(
    adata: AnnData,
    n_top_genes: int = 1200,
    n_bins: int = 51,
) -> AnnData:
    """
    Preprocess the Norman AnnData with the same pipeline as Tutorial_Integration.

    Steps: filter genes, normalize total, log1p, select HVGs, value binning.

    Args:
        adata: Raw Norman AnnData.
        n_top_genes: Number of highly variable genes to keep.
        n_bins: Number of bins for value binning.

    Returns:
        Preprocessed AnnData with layers["X_binned"].
    """
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=n_top_genes,
        hvg_flavor="seurat_v3",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")
    return adata


def filter_genes_to_vocab(
    adata: AnnData,
    vocab: GeneVocab,
    special_tokens: Optional[List[str]] = None,
    case_sensitive: bool = False,
) -> AnnData:
    """
    Keep only genes in the data that exist in the provided vocabulary.
    Uses case-insensitive matching by default to handle differing gene symbol
    conventions (e.g., "Xkr4" vs "XKR4").

    Args:
        adata: AnnData (genes in .var).
        vocab: scGPT GeneVocab.
        special_tokens: List of special tokens.
        case_sensitive: If True, use exact (case-sensitive) matching.

    Returns:
        Filtered AnnData with only matching genes, where gene names in
        adata.var["gene_name"] are updated to match the vocabulary casing.
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<cls>", "<eoc>"]

    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var.index.tolist()

    if not case_sensitive:
        # Build case-insensitive mapping: UPPER -> original vocab key
        vocab_upper = {}
        for token in vocab:
            if token not in special_tokens:
                vocab_upper[token.upper()] = token

        # Map gene names: check if uppercase version exists in vocab
        mapped_names = []
        matched_mask = []
        for gene in adata.var["gene_name"]:
            up = gene.upper()
            if up in vocab_upper:
                mapped_names.append(vocab_upper[up])
                matched_mask.append(True)
            else:
                mapped_names.append(gene)
                matched_mask.append(False)

        n_match = sum(matched_mask)
        logger.info(
            f"Norman matched {n_match}/{len(adata.var)} genes "
            f"in vocabulary of size {len(vocab)} (case-insensitive)."
        )

        if n_match == 0:
            logger.warning("No gene overlap between Norman data and scGPT vocabulary!")
            return adata[:, []]

        # Update gene_name to use vocab casing so downstream code matches correctly
        adata.var["gene_name"] = mapped_names
        adata = adata[:, matched_mask].copy()
        return adata
    else:
        # Original case-sensitive matching
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        n_match = np.sum(adata.var["id_in_vocab"] >= 0)
        logger.info(
            f"Norman matched {n_match}/{len(adata.var)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

        if n_match == 0:
            logger.warning("No gene overlap between Norman data and scGPT vocabulary!")
            return adata[:, []]

        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        return adata


class NormanDataset(Dataset):
    """
    PyTorch Dataset wrapping the Norman AnnData.

    Each sample returns:
        - gene_ids: tokenized gene indices
        - values: binned expression values (masked or unmasked)
        - target_values: original binned values
        - batch_labels: batch labels
        - perturbation_id: integer perturbation label
    """

    def __init__(
        self,
        adata: AnnData,
        gene_ids: np.ndarray,
        vocab: GeneVocab,
        max_len: int,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        mask_ratio: float = 0.0,
        mask_value: int = -1,
        include_zero_gene: bool = True,
        append_cls: bool = True,
    ):
        self.adata = adata
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.include_zero_gene = include_zero_gene
        self.append_cls = append_cls

        input_layer_key = "X_binned"
        self.counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        self.batch_ids = adata.obs["batch_id"].values.astype(np.int64)
        if "perturbation_id" in adata.obs:
            self.perturbation_ids = adata.obs["perturbation_id"].values.astype(np.int64)
        else:
            pert_categories = adata.obs["perturbation"].astype("category")
            self.perturbation_ids = pert_categories.cat.codes.values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.counts[idx]
        tokenized = tokenize_and_pad_batch(
            row[np.newaxis, :],
            self.gene_ids,
            max_len=self.max_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene,
        )
        values = tokenized["values"].squeeze(0)
        genes = tokenized["genes"].squeeze(0)

        # Apply masking
        if self.mask_ratio > 0:
            masked_vals = random_mask_value(
                values.unsqueeze(0).numpy(),
                mask_ratio=self.mask_ratio,
                mask_value=self.mask_value,
                pad_value=self.pad_value,
            ).squeeze(0)
        else:
            masked_vals = values.clone()

        return {
            "gene_ids": genes,
            "values": masked_vals,
            "target_values": values,
            "batch_labels": torch.tensor(self.batch_ids[idx], dtype=torch.long),
            "perturbation_id": torch.tensor(self.perturbation_ids[idx], dtype=torch.long),
        }


def prepare_norman_dataloader(
    adata: AnnData,
    gene_ids: np.ndarray,
    vocab: GeneVocab,
    max_len: int,
    batch_size: int = 64,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    mask_ratio: float = 0.0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the Norman perturbation dataset.
    """
    dataset = NormanDataset(
        adata=adata,
        gene_ids=gene_ids,
        vocab=vocab,
        max_len=max_len,
        pad_token=pad_token,
        pad_value=pad_value,
        mask_ratio=mask_ratio,
        include_zero_gene=True,
        append_cls=True,
    )

    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = adata.obs["batch_id"].values
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return loader

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader