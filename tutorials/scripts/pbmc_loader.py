"""
PBMC 10K dataset loader - standalone implementation without scvi dependency.

Loads the PBMC 10K dataset (pbmc8k + pbmc4k) from 10X Genomics cached
data and adds cell-type labels from the scVI-data repository.

Data sources:
  - 10X Genomics: pbmc8k (8K PBMCs) and pbmc4k (4K PBMCs)
  - YosefLab/scVI-data: gene_info.csv and pbmc_metadata.pickle

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

# Set up a simple logger for this module
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class _CompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler that handles pandas version incompatibilities."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("pandas.core.indexes"):
                return pd.Index
            raise


# ---------------------------------------------------------------------------
# Constants - use pre-cached data from another project to avoid re-download
# ---------------------------------------------------------------------------
PRECACHED_DATA_DIR = Path(
    "/inspire/cpfs/project/sais-ai-for-science-code/public/project/scGPT/data/scgpt_integration/data/PBMC_10K"
)

# Default cache in the current project
LOCAL_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "PBMC_10K"

GENE_INFO_URL = "https://github.com/YosefLab/scVI-data/raw/master/gene_info.csv"
METADATA_URL = "https://github.com/YosefLab/scVI-data/raw/master/pbmc_metadata.pickle"

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
        return filepath

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        with open(filepath, "wb") as f:
            f.write(response.read())
    return filepath


def _copy_cached_file(src_dir: Path, dest_dir: Path, filename: str) -> str:
    """Copy a cached file from src_dir to dest_dir if not already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(str(dest_dir), filename)
    if os.path.exists(dest_path):
        return dest_path
    
    src_path = os.path.join(str(src_dir), filename)
    if os.path.exists(src_path):
        import shutil
        shutil.copy2(src_path, dest_path)
        return dest_path
    
    return None


def _download_and_read_10x(
    dataset_name: str,
    cache_dir: Path,
) -> anndata.AnnData:
    """Download a 10X dataset (tar.gz) and read with scanpy."""
    url = PBMCS_URLS[dataset_name]
    filename = f"{dataset_name}_filtered_gene_bc_matrices.tar.gz"
    
    # First try to use pre-cached data
    filepath = _copy_cached_file(PRECACHED_DATA_DIR, cache_dir, filename)
    if filepath is None:
        filepath = _download_file(url, cache_dir, filename)
    
    # Extract
    extract_dir = cache_dir / dataset_name
    if not extract_dir.exists():
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

    adata = sc.read_10x_mtx(mtx_dir)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)
    return adata


def load_pbmc10k(
    save_path: Optional[str] = None,
    force_redo: bool = False,
) -> anndata.AnnData:
    """
    Load the PBMC 10K dataset (pbmc8k + pbmc4k combined with cell-type labels).
    
    This function mirrors `scvi.data.pbmc_dataset()` but does NOT require
    the scvi package. Data is cached locally after the first download.
    
    Uses pre-cached data from a shared project directory when available
    to avoid re-downloading large files.

    Parameters
    ----------
    save_path : str or None
        Directory to cache the downloaded data. Defaults to data/PBMC_10K/.
    force_redo : bool
        If True, re-process even if cached.

    Returns
    -------
    AnnData with:
      - .obs['batch'] : batch ID (0 for pbmc8k, 1 for pbmc4k)
      - .obs['labels'] : numeric cluster labels
      - .obs['str_labels'] : string cell-type labels
      - .obs['celltype'] : cell-type labels (added for scGPT compatibility)
      - .uns['cell_types'] : list of cell-type names
      - .var['n_counts'] : total counts per gene
    """
    if save_path is not None:
        cache_dir = Path(save_path)
    else:
        cache_dir = LOCAL_CACHE_DIR

    cache_file = cache_dir / "pbmc10k.h5ad"

    if cache_file.exists() and not force_redo:
        adata = anndata.read_h5ad(str(cache_file))
        return adata

    # ------------------------------------------------------------------
    # Download/copy metadata files
    # ------------------------------------------------------------------
    meta_cache = cache_dir / "metadata"
    meta_cache.mkdir(parents=True, exist_ok=True)

    # Try pre-cached metadata first
    gene_info_path = _copy_cached_file(PRECACHED_DATA_DIR, meta_cache, "gene_info_pbmc.csv")
    if gene_info_path is None:
        gene_info_path = _download_file(GENE_INFO_URL, meta_cache, "gene_info.csv")
    
    metadata_path = _copy_cached_file(PRECACHED_DATA_DIR, meta_cache, "pbmc_metadata.pickle")
    if metadata_path is None:
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
    # Filter genes based on DE metadata (match using gene symbols, not ENSG IDs)
    # The 10X data uses gene symbols as var_names, so we match against the GS column
    # ------------------------------------------------------------------
    gene_symbols_metadata = list(de_metadata["GS"].dropna().unique())
    genes_in_data = set(adata.var_names)
    genes_to_keep = [g for g in gene_symbols_metadata if g in genes_in_data]
    if len(genes_to_keep) == 0:
        # Fallback: try matching ENSG IDs directly against var_names
        logger.warning("No gene symbols matched, trying ENSG ID matching...")
        genes_to_keep = []
        for ensg_id in de_metadata["ENSG"].values:
            if ensg_id in adata.var_names:
                genes_to_keep.append(ensg_id)
    if len(genes_to_keep) == 0:
        # Last resort: keep all genes
        logger.warning("No genes matched metadata, keeping all genes")
        genes_to_keep = list(adata.var_names)
    logger.info(f"Filtered to {len(genes_to_keep)} genes matching DE metadata")
    adata = adata[:, genes_to_keep].copy()

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
    
    # Add celltype column for scGPT compatibility (used in preprocessing)
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")

    adata.var["n_counts"] = np.squeeze(np.asarray(np.sum(adata.X, axis=0)))
    
    # Add gene_symbols column (matching scvi's pbmc_dataset behavior)
    # Map ENSG IDs to gene symbols using de_metadata
    ensg_to_symbol = dict(zip(de_metadata["GS"].values, de_metadata["ENSG"].values))
    adata.var["gene_symbols"] = [ensg_to_symbol.get(g, g) for g in adata.var_names]
    adata.var_names_make_unique()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)
    adata.write(str(cache_file))

    return adata