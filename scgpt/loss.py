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
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor,
    eps: float = 1e-3, max_val: float = 1e4,
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.

    Improved with:
      - Larger epsilon (1e-3) to avoid division by near-zero values
      - Clipping to prevent extreme values from dominating
      - Support for zero targets: if target is zero, use absolute error
        instead of division by zero

    Args:
        input: predicted values
        target: ground truth values
        mask: boolean mask of positions to evaluate
        eps: small constant to avoid division by zero
        max_val: maximum allowed relative error value
    """
    assert mask.any()
    inp = input[mask]
    tgt = target[mask]

    # For zero targets, use absolute error instead of relative
    zero_target = (tgt.abs() < eps)
    if zero_target.any():
        # Use absolute error for zero targets
        abs_error = torch.abs(inp - tgt)
        rel_error = torch.abs(inp - tgt) / (tgt.abs() + eps)
        # For zero targets, scale absolute error by a sensible factor
        rel_error = torch.where(zero_target, abs_error / eps, rel_error)
    else:
        rel_error = torch.abs(inp - tgt) / (tgt.abs() + eps)

    # Clamp extreme values to prevent outlier domination
    rel_error = torch.clamp(rel_error, max=max_val)
    return rel_error.mean()


def masked_huber_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Compute the masked Huber loss between input and target.

    Huber loss is less sensitive to outliers than MSE, making it
    more robust for noisy single-cell expression data.

    Args:
        input: predicted values
        target: ground truth values
        mask: boolean mask of positions to evaluate
        delta: threshold at which to switch from L2 to L1 loss
    """
    mask = mask.float()
    diff = (input - target).abs()
    loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    return (loss * mask).sum() / mask.sum()


def cell_type_contrastive_loss(
    cell_emb: torch.Tensor,
    celltype_labels: torch.Tensor,
    batch_labels: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    batch_weight: float = 0.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cell-type Contrastive (CTC) loss for cross-batch alignment (v2 - Improved).

    Pulls cells of the same type closer together while pushing cells of
    different types apart. Uses InfoNCE-style supervised contrastive loss.

    Key improvements over v1:
      - Higher temperature (0.5 vs 0.2) for smoother gradient dynamics and
        better positive pair discovery in high-dimensional space
      - Higher batch_weight (0.7 vs 0.5) to better preserve cross-batch
        cell-type relationships while maintaining batch correction
      - Self-pair weighting: when same cell has same type AND same batch,
        weight is increased to strengthen within-batch same-type alignment
      - Adaptive temperature scaling: if batch_labels are provided, the
        effective temperature increases slightly to prevent over-strong
        negative gradient for cross-type pairs in different batches
      - Numerically stable with improved LogSumExp and gradient gating

    Args:
        cell_emb: Tensor, shape [batch_size, d_model]
        celltype_labels: Tensor, shape [batch_size], cell type IDs
        batch_labels: Tensor, shape [batch_size], optional batch IDs
        temperature: temperature scaling for the contrastive loss
        batch_weight: weight for cross-batch positive pairs (0.0-1.0)
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
        # Cross-batch positive pairs get reduced weight
        pos_weights[cross_batch_same_type] = batch_weight
        # Within-batch same-type pairs get full weight
        pos_weights[pos_mask & same_batch] = 1.0
        pos_weights[~pos_mask] = 0.0
    else:
        pos_weights = pos_mask.float()

    # Adaptive temperature: increase if batch labels exist (more diversity)
    effective_temp = temperature * (1.0 + 0.1 * (batch_labels is not None))

    # Numerically stable InfoNCE loss using LogSumExp trick
    cos_sim_scaled = cos_sim / effective_temp

    # Compute log denominator: log(sum_j exp(sim_ij/T)) for j != i
    sim_max, _ = cos_sim_scaled.max(dim=1, keepdim=True)
    exp_sim = torch.exp(cos_sim_scaled - sim_max.detach()) * (~mask_diag).float()
    log_sum_exp = sim_max.detach().squeeze(1) + torch.log(exp_sim.sum(dim=1) + eps)

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
