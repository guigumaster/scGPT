"""
PBMC 10K dataset loader - scvi-free alternative.

Loads the PBMC 10K dataset (pbmc8k + pbmc4k) used in the scGPT integration
tutorial without requiring the scvi package (which depends on jaxlib).

Data sources:
  - 10X Genomics: pbmc8k (4K PBMCs) and pbmc4k (8K PBMCs)
  - YosefLab/scVI-data: gene_info.csv and pbmc_metadata.pickle for cell-type labels

Reference:
  scvi.data.pbmc_dataset()
"""

import os
import pickle
import tarfile
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from scgpt import logger


class _CompatibilityUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles pandas version incompatibilities.
    When a module is not found (e.g., `pandas.core.indexes.numeric`
    renamed in newer pandas), it falls back to a generic Index.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Handle pandas index compatibility
            if module.startswith("pandas.core.indexes"):
                return pd.Index
            raise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PBMC_CACHE_DIR = Path(os.path.expanduser("~/.cache/scgpt/pbmc10k"))
PBMC_CACHE_FILE = PBMC_CACHE_DIR / "pbmc10k.h5ad"

GENE_INFO_URL = "https://github.com/YosefLab/scVI-data/raw/master/gene_info.csv"
METADATA_URL = "https://github.com/YosefLab/scVI-data/raw/master/pbmc_metadata.pickle"

# 10X URLs for pbmc8k (v2.1.0) and pbmc4k (v2.1.0)
PBMCS_URLS = {
    "pbmc8k": (
        "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc8k/"
        "pbmc8k_filtered_gene_bc_matrices.tar.gz"
    ),
    "pbmc4k": (
        "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/"
        "pbmc4k_filtered_gene_bc_matrices.tar.gz"
    ),
}


def _download_file(url: str, save_path: Path, filename: str) -> str:
    """Download a file from url to save_path/filename."""
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(str(save_path), filename)
    if os.path.exists(filepath):
        logger.info(f"[PBMC] Already cached: {filepath}")
        return filepath

    logger.info(f"[PBMC] Downloading {url} to {filepath}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        with open(filepath, "wb") as f:
            f.write(response.read())
    logger.info(f"[PBMC] Downloaded: {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")
    return filepath


def _download_and_read_10x(
    dataset_name: str,
    cache_dir: Path,
) -> anndata.AnnData:
    """Download a 10X dataset (tar.gz) and read with scanpy."""
    url = PBMCS_URLS[dataset_name]
    filename = f"{dataset_name}_filtered_gene_bc_matrices.tar.gz"
    filepath = _download_file(url, cache_dir, filename)

    # Extract
    extract_dir = cache_dir / dataset_name
    if not extract_dir.exists():
        logger.info(f"[PBMC] Extracting {filepath}")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=str(extract_dir))

    # Find the matrix.mtx directory
    mtx_dir = None
    for root, dirs, files in os.walk(str(extract_dir)):
        if "matrix.mtx" in files:
            mtx_dir = root
            break

    if mtx_dir is None:
        raise FileNotFoundError(f"Could not find matrix.mtx in {extract_dir}")

    adata = sc.read_10x_mtx(mtx_dir, var_names='gene_ids')
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)
    logger.info(f"[PBMC] Loaded {dataset_name}: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def load_pbmc10k(
    save_path: Optional[str] = None,
    force_redo: bool = False,
) -> anndata.AnnData:
    """
    Load the PBMC 10K dataset (pbmc8k + pbmc4k combined with cell-type labels).

    This function mirrors `scvi.data.pbmc_dataset()` but does NOT require
    the scvi package. Data is cached locally after the first download.

    Parameters
    ----------
    save_path : str or None
        Directory to cache the downloaded data. Defaults to ~/.cache/scgpt/pbmc10k/.
    force_redo : bool
        If True, re-download and re-process even if cached.

    Returns
    -------
    AnnData with:
      - .obs['batch'] : batch ID (0 for pbmc8k, 1 for pbmc4k)
      - .obs['labels'] : numeric cluster labels
      - .obs['str_labels'] : string cell-type labels
      - .uns['cell_types'] : list of cell-type names
      - .var['n_counts'] : total counts per gene
    """
    if save_path is not None:
        cache_dir = Path(save_path)
    else:
        cache_dir = PBMC_CACHE_DIR

    cache_file = cache_dir / "pbmc10k.h5ad"

    if cache_file.exists() and not force_redo:
        logger.info(f"[PBMC] Loading cached dataset from {cache_file}")
        adata = anndata.read_h5ad(str(cache_file))
        logger.info(f"[PBMC] Cached dataset: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata

    # ------------------------------------------------------------------
    # Download metadata files
    # ------------------------------------------------------------------
    meta_cache = cache_dir / "metadata"
    meta_cache.mkdir(parents=True, exist_ok=True)

    gene_info_path = _download_file(GENE_INFO_URL, meta_cache, "gene_info.csv")
    metadata_path = _download_file(METADATA_URL, meta_cache, "pbmc_metadata.pickle")

    de_metadata = pd.read_csv(gene_info_path, sep=",")
    with open(metadata_path, "rb") as f:
        pbmc_metadata = _CompatibilityUnpickler(f).load()

    # ------------------------------------------------------------------
    # Download and load 10X data
    # ------------------------------------------------------------------
    pbmc8k = _download_and_read_10x("pbmc8k", cache_dir)
    pbmc4k = _download_and_read_10x("pbmc4k", cache_dir)

    barcodes = np.concatenate((pbmc8k.obs_names, pbmc4k.obs_names))
    adata = pbmc8k.concatenate(pbmc4k)
    adata.obs_names = barcodes

    # ------------------------------------------------------------------
    # Filter cells based on metadata barcodes
    # ------------------------------------------------------------------
    dict_barcodes = dict(zip(barcodes, np.arange(len(barcodes))))
    barcodes_metadata = pbmc_metadata["barcodes"].index.values.ravel().astype(str)
    subset_cells = []
    for barcode in barcodes_metadata:
        if barcode in dict_barcodes:
            subset_cells.append(dict_barcodes[barcode])
    adata = adata[np.asarray(subset_cells), :].copy()

    idx_metadata = np.asarray(
        [not barcode.endswith("11") for barcode in barcodes_metadata], dtype=bool
    )

    # ------------------------------------------------------------------
    # Filter genes based on DE metadata
    # ------------------------------------------------------------------
    genes_to_keep = list(de_metadata["ENSG"].values)
    difference = list(set(genes_to_keep).difference(set(adata.var_names)))
    for gene in difference:
        genes_to_keep.remove(gene)
    adata = adata[:, genes_to_keep].copy()

    # ------------------------------------------------------------------
    # Add gene_symbols column to var (used by Tutorial_Integration.py)
    # ------------------------------------------------------------------
    # The CSV column is 'GS' (Gene Symbol), not 'gene_symbols'
    gene_symbol_map = dict(zip(de_metadata["ENSG"], de_metadata["GS"]))
    adata.var["gene_symbols"] = [
        gene_symbol_map.get(gene, gene) for gene in adata.var_names
    ]

    # ------------------------------------------------------------------
    # Attach metadata
    # ------------------------------------------------------------------
    design = pbmc_metadata["design"][idx_metadata]
    raw_qc = pbmc_metadata["raw_qc"][idx_metadata]
    normalized_qc = pbmc_metadata["normalized_qc"][idx_metadata]

    design.index = adata.obs_names
    raw_qc.index = adata.obs_names
    normalized_qc.index = adata.obs_names

    adata.obs["batch"] = adata.obs["batch"].astype(np.int64)
    adata.obsm["design"] = design
    adata.obsm["raw_qc"] = raw_qc
    adata.obsm["normalized_qc"] = normalized_qc
    adata.obsm["qc_pc"] = pbmc_metadata["qc_pc"][idx_metadata]

    labels = pbmc_metadata["clusters"][idx_metadata]
    cell_types = pbmc_metadata["list_clusters"]
    adata.obs["labels"] = labels
    adata.uns["cell_types"] = cell_types
    adata.obs["str_labels"] = [cell_types[i] for i in labels]

    adata.var["n_counts"] = np.squeeze(np.asarray(np.sum(adata.X, axis=0)))

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)
    adata.write(str(cache_file))
    logger.info(f"[PBMC] Cached dataset to {cache_file}")

    logger.info(
        f"[PBMC] PBMC 10K loaded: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{len(cell_types)} cell types"
    )
    return adata