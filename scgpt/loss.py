import torch
import torch.nn.functional as F
import numpy as np


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


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
    base_temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss.
    Pulls together cell embeddings of the same cell type and pushes apart
    embeddings of different cell types.

    Reference: https://arxiv.org/abs/2004.11362

    Args:
        features: hidden features of shape [batch_size, feature_dim]
        labels: ground truth labels of shape [batch_size]
        temperature: temperature scaling parameter
        base_temperature: base temperature for scaling

    Returns:
        loss: supervised contrastive loss scalar
    """
    device = features.device
    batch_size = features.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create labels mask: 1 if same label, 0 otherwise
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Remove diagonal (self-similarity)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # Compute log probabilities
    exp_logits = torch.exp(similarity_matrix) * logits_mask

    # Log probability of positive pairs
    log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-likelihood over positive pairs
    pos_count = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    loss = - (temperature / base_temperature) * mean_log_prob_pos
    # Only consider samples that have at least one positive pair
    loss = loss[pos_count > 0].mean()

    return loss


def compute_gene_f_statistics(
    data: np.ndarray,
    celltype_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute ANOVA F-statistics for each gene across cell types.
    Measures how well each gene distinguishes between different cell types.
    Higher F-statistic = more cell-type-specific expression pattern.

    Args:
        data: gene expression data of shape [n_cells, n_genes]
        celltype_labels: cell type labels of shape [n_cells]

    Returns:
        f_stats: F-statistics for each gene, shape [n_genes]
    """
    unique_labels = np.unique(celltype_labels)
    n_groups = len(unique_labels)
    n_cells = data.shape[0]

    if n_groups < 2:
        return np.ones(data.shape[1])

    f_stats = np.zeros(data.shape[1])
    grand_mean = np.mean(data, axis=0)

    for g in range(data.shape[1]):
        gene_expr = data[:, g]

        ss_between = 0.0
        ss_within = 0.0

        for label in unique_labels:
            group_mask = celltype_labels == label
            group_expr = gene_expr[group_mask]
            if len(group_expr) == 0:
                continue
            group_mean = np.mean(group_expr)
            n_group = len(group_expr)

            ss_between += n_group * (group_mean - grand_mean[g]) ** 2
            ss_within += np.sum((group_expr - group_mean) ** 2)

        df_between = max(n_groups - 1, 1)
        df_within = max(n_cells - n_groups, 1)

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        f_stats[g] = ms_between / max(ms_within, 1e-10)

    # Handle NaN/Inf values
    f_stats = np.nan_to_num(f_stats, nan=0.0, posinf=0.0)

    return f_stats
