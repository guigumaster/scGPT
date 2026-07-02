"""
On-the-fly Masking Datasets for PCT-AIM tasks
==============================================
These datasets store unmasked tokenized data and apply task-adaptive random masking
per sample during __getitem__, eliminating the need to re-mask the entire dataset
every epoch. This significantly reduces per-epoch overhead and speeds up training.

Each task (perturbation, integration, large_perturbation) has its own subclass
that implements the appropriate masking strategy.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Union


class MaskedSeqDataset(Dataset):
    """
    Base dataset that stores pre-tokenized data and applies on-the-fly masking.
    
    Each call to __getitem__ returns a sample with a fresh random mask,
    which means different positions are masked in each epoch automatically.
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        mask_ratio: float = 0.35,
        mask_value: int = -1,
        pad_value: int = -2,
        task_type: str = "perturbation",
        perturbed_genes_mask: Optional[torch.Tensor] = None,
        gene_importance: Optional[torch.Tensor] = None,
    ):
        # Store gene_ids, target_values, batch_labels, pert_labels etc.
        self.gene_ids = data["gene_ids"]
        self.target_values = data["target_values"]
        self.batch_labels = data["batch_labels"]
        
        # Optional fields
        self.pert_labels = data.get("pert_labels", None)
        self.pert_multilabel = data.get("pert_multilabel", None)
        self.celltype_labels = data.get("celltype_labels", None)

        # Masking parameters
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.task_type = task_type
        self.perturbed_genes_mask = perturbed_genes_mask
        self.gene_importance = gene_importance

    def __len__(self):
        return self.gene_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "gene_ids": self.gene_ids[idx],
            "target_values": self.target_values[idx],
            "batch_labels": self.batch_labels[idx],
        }
        if self.pert_labels is not None:
            item["pert_labels"] = self.pert_labels[idx]
        if self.pert_multilabel is not None:
            item["pert_multilabel"] = self.pert_multilabel[idx]
        if self.celltype_labels is not None:
            item["celltype_labels"] = self.celltype_labels[idx]
        
        # Apply on-the-fly masking to the values
        item["values"] = self._apply_mask(self.target_values[idx])
        
        return item

    def _apply_mask(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply task-adaptive random masking to a sample.
        Fully vectorized per-sample masking.
        """
        seq_len = values.shape[0]
        device = values.device

        # Determine per-gene mask probabilities
        if self.task_type == "perturbation":
            base_prob = min(self.mask_ratio, 0.4)
            gene_prob = torch.full((seq_len,), base_prob, device=device)
            if self.perturbed_genes_mask is not None:
                p_mask = self.perturbed_genes_mask.to(device)
                mask_len = min(len(p_mask), seq_len)
                gene_prob[:mask_len] = torch.where(
                    p_mask[:mask_len],
                    torch.tensor(min(base_prob + 0.3, 0.85), device=device),
                    gene_prob[:mask_len],
                )
        elif self.task_type == "large_perturbation":
            base_prob = min(self.mask_ratio, 0.35)
            gene_prob = torch.full((seq_len,), base_prob, device=device)
            if self.perturbed_genes_mask is not None:
                p_mask = self.perturbed_genes_mask.to(device)
                mask_len = min(len(p_mask), seq_len)
                gene_prob[:mask_len] = torch.where(
                    p_mask[:mask_len],
                    torch.tensor(min(base_prob + 0.25, 0.8), device=device),
                    gene_prob[:mask_len],
                )
        elif self.task_type == "integration":
            base_prob = min(self.mask_ratio, 0.35)
            gene_prob = torch.full((seq_len,), base_prob, device=device)
            if self.gene_importance is not None:
                g_imp = self.gene_importance.to(device)
                imp_len = min(len(g_imp), seq_len)
                imp = g_imp[:imp_len]
                imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
                gene_prob[:imp_len] = torch.clamp(base_prob + 0.2 * imp_norm, 0.1, 0.85)
        else:
            gene_prob = torch.full((seq_len,), self.mask_ratio, device=device)

        # CLS token (position 0) is never masked
        gene_prob[0] = 0.0

        # Generate mask
        masked = values.clone()
        rand_vals = torch.rand(seq_len, device=device)
        pad_mask = (values == self.pad_value)
        mask = (rand_vals < gene_prob) & ~pad_mask

        # Ensure at least 1 position is masked (if not all padding)
        if not mask.any() and not pad_mask.all():
            available = torch.where(~pad_mask)[0]
            available = available[available != 0]  # exclude CLS
            if len(available) > 0:
                chosen = available[torch.randint(0, len(available), (1,))]
                mask[chosen] = True

        masked[mask] = self.mask_value
        return masked.float()