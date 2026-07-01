import torch
import torch.nn.functional as F


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
    Uses robust handling: only computes on positions where target > 0,
    and clips the ratio to avoid numerical explosion from near-zero targets.
    """
    assert mask.any()
    # Only compute on positions where target > 0 to avoid division by near-zero
    positive_mask = mask & (target > 0.5)
    if not positive_mask.any():
        return torch.tensor(0.0, device=input.device)
    loss = torch.abs(input[positive_mask] - target[positive_mask]) / (target[positive_mask] + 1e-6)
    # Clip to prevent extreme values from dominating
    loss = torch.clamp(loss, max=1e4)
    return loss.mean()
