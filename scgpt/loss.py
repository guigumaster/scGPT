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
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


def prototype_contrastive_loss(
    cell_emb: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Prototype contrastive loss (ProtoNCE).
    Pulls cell embeddings toward their class prototype, pushes away from others.

    Args:
        cell_emb: (batch, d_model) normalized cell embeddings
        prototypes: (n_cls, d_model) prototype vectors
        labels: (batch,) cell type labels
        temperature: temperature for scaling logits

    Returns:
        loss scalar
    """
    cell_emb = F.normalize(cell_emb, p=2, dim=1)
    proto = F.normalize(prototypes, p=2, dim=1)

    # logits: (batch, n_cls)
    logits = torch.mm(cell_emb, proto.t()) / temperature

    loss = F.cross_entropy(logits, labels)
    return loss
