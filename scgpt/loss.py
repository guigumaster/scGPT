import torch
import torch.nn.functional as F
from typing import Optional


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


def cell_type_contrastive_loss(
    cell_emb: torch.Tensor,
    celltype_labels: torch.Tensor,
    batch_labels: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    batch_weight: float = 0.3,
) -> torch.Tensor:
    """
    Cell-type Contrastive (CTC) loss for cross-batch alignment.
    
    Pulls cells of the same type closer together while pushing cells of
    different types apart. When batch_labels are provided, also applies
    a weaker pull for cells from different batches but same cell type,
    enhancing cross-batch alignment.
    
    Args:
        cell_emb: Tensor, shape [batch_size, d_model]
        celltype_labels: Tensor, shape [batch_size], cell type IDs
        batch_labels: Tensor, shape [batch_size], optional batch IDs
        temperature: temperature scaling for the contrastive loss
        batch_weight: weight for cross-batch positive pairs
    
    Returns:
        loss: scalar tensor
    """
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
    
    mask_diag = torch.eye(cos_sim.size(0), device=cos_sim.device).bool()
    ctype = celltype_labels
    
    # Same cell type mask
    pos_mask = ctype.unsqueeze(1) == ctype.unsqueeze(0)
    pos_mask = pos_mask & ~mask_diag
    
    # Scale: same cell type + same batch gets higher weight
    if batch_labels is not None:
        batch_same = batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0)
        # Cross-batch but same cell type pairs get reduced weight
        cross_batch_same_type = pos_mask & ~batch_same
        pos_mask = pos_mask.float()
        pos_mask[cross_batch_same_type] = batch_weight
    
    # Convert to float for weighted sum if batch_labels provided, otherwise bool mask
    pos_mask_float = pos_mask.float() if batch_labels is not None else pos_mask.float()
    pos_count = pos_mask_float.sum(dim=1)
    
    cos_sim_scaled = cos_sim / temperature
    exp_sim = torch.exp(cos_sim_scaled) * (~mask_diag).float()
    log_prob = cos_sim_scaled - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    
    pos_loss = -(log_prob * pos_mask_float).sum(dim=1) / (pos_count + 1e-8)
    
    return pos_loss.mean()
