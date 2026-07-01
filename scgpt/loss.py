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
    temperature: float = 0.2,
    batch_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cell-type Contrastive (CTC) loss for cross-batch alignment.

    Pulls cells of the same type closer together while pushing cells of
    different types apart. Uses InfoNCE-style supervised contrastive loss.

    When batch_labels are provided, cross-batch same-type pairs get
    a configurable weight (batch_weight < 1.0) to balance batch correction
    with biological conservation.

    Uses improved temperature=0.2 for smoother gradient dynamics and
    numerically stable log-softmax computation (LogSumExp trick).

    Args:
        cell_emb: Tensor, shape [batch_size, d_model]
        celltype_labels: Tensor, shape [batch_size], cell type IDs
        batch_labels: Tensor, shape [batch_size], optional batch IDs
        temperature: temperature scaling for the contrastive loss
        batch_weight: weight for cross-batch positive pairs
        eps: small constant for numerical stability

    Returns:
        loss: scalar tensor
    """
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())

    batch_size = cos_sim.size(0)
    mask_diag = torch.eye(batch_size, device=cos_sim.device, dtype=torch.bool)
    
    # Get positive mask (same cell type, excluding self)
    ctype = celltype_labels
    pos_mask = (ctype.unsqueeze(1) == ctype.unsqueeze(0)) & ~mask_diag

    # Compute weights: cross-batch same-type pairs get batch_weight
    if batch_labels is not None:
        same_batch = batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0)
        cross_batch_same_type = pos_mask & ~same_batch
        pos_weights = torch.ones_like(cos_sim, dtype=torch.float)
        pos_weights[cross_batch_same_type] = batch_weight
        pos_weights[~pos_mask] = 0.0
    else:
        pos_weights = pos_mask.float()

    # Numerically stable InfoNCE loss using LogSumExp trick
    # log_prob = cos_sim / T - log(sum(exp(cos_sim / T)))
    cos_sim_scaled = cos_sim / temperature
    
    # Compute log denominator: log(sum_j exp(sim_ij/T)) for j != i
    # Use max trick for numerical stability: log(sum exp(x)) = max + log(sum exp(x - max))
    sim_max, _ = cos_sim_scaled.max(dim=1, keepdim=True)
    exp_sim = torch.exp(cos_sim_scaled - sim_max.detach()) * (~mask_diag).float()
    log_sum_exp = sim_max.detach().squeeze(1) + torch.log(exp_sim.sum(dim=1) + eps)
    
    log_prob = cos_sim_scaled.diagonal(dim1=0, dim2=1)  # Not used directly
    # Per-anchor supervised contrastive: log(exp(sim_ij/T) / sum_k exp(sim_ik/T))
    log_prob = cos_sim_scaled - log_sum_exp.unsqueeze(1)

    # Weighted sum over positive pairs
    pos_count = pos_weights.sum(dim=1)
    pos_loss = -(log_prob * pos_weights).sum(dim=1) / (pos_count + eps)
    
    # Only include anchors that have at least one positive pair
    valid = pos_count > 0
    if valid.any():
        return pos_loss[valid].mean()
    else:
        return torch.tensor(0.0, device=cell_emb.device, requires_grad=True)
