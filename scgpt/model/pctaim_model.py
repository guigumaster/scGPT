"""
PCT-AIM: Perturbation-Conditioned Cross-Modal Transformer with Adaptive Importance Masking
============================================================================================

Three tightly coupled mechanisms:
  1) Perturbation Condition Encoder  – embed perturbation labels and fuse into gene token
     representations so the model explicitly perceives the perturbation state.
  2) Cross-Modal Cross-Attention Layer – establish explicit alignment between RNA and
     Protein (or other) modalities for superior multi-omic integration.
  3) Task-Adaptive Importance Masking – dynamically adjust mask probabilities per gene
     based on the task type (perturbation prediction, multi-omic integration, large-scale
     perturbation).

All three modules share a unified optimisation across the three core tasks, improving
perturbation prediction accuracy, cell-clustering consistency (ARI / NMI), batch-effect
correction (PCR_batch), and generated-expression correlation.
"""

import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange

from .flash_attn_compat import FlashMHA, flash_attn_available
from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse

# ---------------------------------------------------------------------------
#  Utility modules (adapted from model.py / multiomic_model.py)
# ---------------------------------------------------------------------------

class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,
                                      padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,
                                      padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,
                                      padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Enhanced similarity module with learnable temperature for CCE.
    Uses a learnable temperature parameter that starts at a reasonable value
    and is clamped to prevent extreme values that could collapse contrastive learning.
    
    v6 improvement: Lower initial temperature (0.1) for sharper contrastive signals
    at the start of training, leading to better separation of cell types.
    """
    def __init__(self, temp=0.1):
        super().__init__()
        # Learnable temperature with better initialization (lower = sharper)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temp)), requires_grad=True)
        self.cos = nn.CosineSimilarity(dim=-1)

    @property
    def temp(self):
        # Clamp temperature to [0.03, 0.5] for stable contrastive learning
        return torch.clamp(self.log_temp.exp(), min=0.03, max=0.5)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    def __init__(self, d_model: int, explicit_zero_prob: bool = False,
                 use_batch_labels: bool = False, use_mod: bool = False):
        super().__init__()
        d_in = d_model * 2 if (use_batch_labels or use_mod) else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred_value = self.fc(x).squeeze(-1)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3,
                 activation: callable = nn.ReLU):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    def __init__(self, d_model: int, arch_style: str = "inner product",
                 query_activation: nn.Module = nn.Sigmoid,
                 hidden_activation: nn.Module = nn.PReLU,
                 explicit_zero_prob: bool = False,
                 use_batch_labels: bool = False,
                 use_mod: bool = False):
        super().__init__()
        d_in = d_model * 2 if (use_batch_labels or use_mod) else d_model
        d_model_eff = d_model * 2 if (use_batch_labels or use_mod) else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model_eff, d_model_eff)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model_eff, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model_eff, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model_eff, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model_eff + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model_eff, d_model_eff)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model_eff, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")
        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(self, cell_emb: Tensor, gene_embs: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)
            h = self.hidden_activation(self.fc1(torch.cat([cell_emb, query_vecs], dim=2)))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)


class AdversarialDiscriminator(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3,
                 activation: callable = nn.LeakyReLU, reverse_grad: bool = False):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class EnhancedAdversarialDiscriminator(nn.Module):
    """
    Enhanced discriminator for adversarial batch correction with gradient penalty.
    Uses larger capacity, gradient reversal with learnable scaling,
    and spectral normalization for stable training.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 4,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
        lambd: float = 1.0,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        hidden_dims = [d_model * 2, d_model * 2, d_model, d_model // 2]
        for i in range(min(nlayers, len(hidden_dims)) - 1):
            in_dim = d_model if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            linear = nn.Linear(in_dim, out_dim)
            # Apply spectral normalization for stable adversarial training
            try:
                linear = nn.utils.spectral_norm(linear)
            except Exception:
                pass
            self._decoder.append(linear)
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(out_dim))
            self._decoder.append(nn.Dropout(0.1))
        self.out_layer = nn.Linear(hidden_dims[min(nlayers, len(hidden_dims)) - 2], n_cls)
        self.reverse_grad = reverse_grad
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        if self.reverse_grad:
            x = grad_reverse(x, lambd=self.lambd)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

    def compute_gradient_penalty(self, real_emb: Tensor, fake_emb: Tensor) -> Tensor:
        """
        Compute gradient penalty for Lipschitz continuity.
        Encourages the discriminator to have 1-Lipschitz gradients.
        """
        batch_size = real_emb.size(0)
        alpha = torch.rand(batch_size, 1, device=real_emb.device)
        alpha = alpha.expand_as(real_emb)
        interpolates = alpha * real_emb + (1 - alpha) * fake_emb
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class PerturbationPredictor(nn.Module):
    """
    Enhanced multi-label perturbation prediction head v5.
    Predicts which perturbations are present in a cell based on its embedding.
    Uses deeper architecture with residual connections, better regularization,
    and adaptive scaling for improved multi-label prediction.
    """
    def __init__(self, d_model: int, n_perturbations: int, nlayers: int = 4):
        super().__init__()
        # Wider hidden dims for better capacity
        hidden_dim = d_model * 2
        layers = []
        for i in range(nlayers - 1):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = hidden_dim if i < nlayers - 2 else d_model
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.1 if i < nlayers - 2 else 0.05),
            ])
        self.fc = nn.Sequential(*layers)
        # Residual projection if dimensions match
        self.has_residual = (d_model == d_model)  # always true, shortcut identity
        # Additional dropout before final classifier for better regularization
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, n_perturbations)
        # Learnable logit scaling for better multi-label prediction
        self.logit_scale = nn.Parameter(torch.ones(1) * 0.5)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize final classifier with small weights for stable start
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.fc(x)
        if self.has_residual:
            x = x + identity
        x = self.classifier_dropout(x)
        return self.classifier(x) * self.logit_scale

    def compute_confidence(self, x: Tensor) -> Tensor:
        """
        Compute prediction confidence for uncertainty-aware training.
        Returns max probability after sigmoid for multi-label classification.
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs.max(dim=1)[0]


class FastTransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout)

    @staticmethod
    def build_fast_transformer_encoder(d_model, nhead, d_hid, nlayers, dropout):
        from fast_transformers.builders import TransformerEncoderBuilder
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers, n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout, dropout=dropout, activation="gelu",
        )
        return builder.get()

    @staticmethod
    def build_length_mask(src, src_key_padding_mask):
        from fast_transformers.masking import LengthMask
        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings
        return LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

    def forward(self, src, src_key_padding_mask):
        length_mask = self.build_length_mask(src, src_key_padding_mask)
        return self.fast_transformer_encoder(src, length_mask=length_mask)


class FlashTransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None, norm_scheme="post"):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model, num_heads=nhead, batch_first=batch_first,
            attention_dropout=dropout, **factory_kwargs)
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")
        if not src_key_padding_mask.any().item():
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            src_key_padding_mask_ = ~src_key_padding_mask
        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


# ---------------------------------------------------------------------------
#  Mechanism 1: Perturbation Condition Encoder
# ---------------------------------------------------------------------------

class PerturbationConditionEncoder(nn.Module):
    """
    Encodes perturbation labels (e.g. guide identity, condition type) into
    continuous embeddings and fuses them into the gene token representations
    via element-wise addition after a learned projection.

    Supports both:
      - Single perturbation labels per cell (batch_size,)
      - Per-gene perturbation flags (batch_size, seq_len)

    Enhanced with:
      - Deeper MLP for cell-level perturbation with larger hidden dim
      - Better initialization for stable early training
      - Adaptive scaling with learnable temperature
    """

    def __init__(self, num_perturbations: int, d_model: int,
                 padding_idx: int = 0, per_gene: bool = True):
        super().__init__()
        self.per_gene = per_gene
        self.d_model = d_model
        if per_gene:
            # Per-gene perturbation flag encoder (0/1/2: no pert / pert / padding)
            self.pert_embedding = nn.Embedding(3, d_model, padding_idx=padding_idx)
        else:
            # Global perturbation label encoder with higher capacity
            self.pert_embedding = nn.Embedding(num_perturbations, d_model, padding_idx=0)
            # Deeper MLP to project perturbation embedding to richer representation
            self.pert_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
        # Learnable scale factor to balance perturbation signal vs gene expression
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        # Temperature for scaling perturbation influence
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, pert_labels: Tensor) -> Tensor:
        """
        Args:
            pert_labels: (batch_size,) or (batch_size, seq_len) – perturbation
                         labels / flags.

        Returns:
            pert_emb: (batch_size, d_model) or (batch_size, seq_len, d_model)
        """
        pert_emb = self.pert_embedding(pert_labels)
        if not self.per_gene and pert_emb.dim() == 2:
            # Apply projection MLP for cell-level perturbation embeddings
            pert_emb = self.pert_proj(pert_emb)
        return pert_emb * self.scale * self.temperature


# ---------------------------------------------------------------------------
#  Mechanism 2: Cross-Modal Cross-Attention Layer
# ---------------------------------------------------------------------------

class CrossModalCrossAttention(nn.Module):
    """
    Establishes explicit alignment between RNA and Protein (or other) modalities.

    Uses a cross-attention mechanism where:
      - Query  = RNA modality tokens
      - Key/Value = Protein (or second modality) tokens

    Enhanced version with:
      - Multi-head cross-attention with proper Pre-LN
      - Learnable gating with sigmoid normalization
      - Dropout on attention output
      - LayerNorm on output
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable gating for adaptive fusion (sigmoid-normalized to [0,1])
        self.gate_logit = nn.Parameter(torch.zeros(1))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, rna_repr: Tensor, prot_repr: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            rna_repr:  (batch, seq_len_rna, d_model)
            prot_repr: (batch, seq_len_prot, d_model)
            key_padding_mask: (batch, seq_len_prot) – True for padding in prot modality.

        Returns:
            output: (batch, seq_len_rna, d_model) – RNA representations updated
                    with protein context via residual connection.
        """
        # Pre-LN: normalize before attention
        q = self.norm_q(rna_repr)
        kv = self.norm_kv(prot_repr)

        # Convert mask format: True=padding (PyTorch) -> need for nn.MultiheadAttention
        # nn.MultiheadAttention key_padding_mask: True=padding=ignored
        attn_out, attn_weights = self.cross_attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,  # True=padding
        )
        # Gated residual connection
        gated_update = self.gate * self.dropout(attn_out)
        output = rna_repr + gated_update
        output = self.norm_out(output)
        return output


class CrossModalFusionLayer(nn.Module):
    """
    A fusion layer that combines RNA and Protein representations through
    bidirectional cross-attention between modalities.

    Enhanced version with:
      - Deeper fusion network with residual connections
      - Learnable gating with element-wise gates per token
      - Modality-specific output projections
      - Multi-layer fusion for deeper alignment
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.rna_to_prot = CrossModalCrossAttention(d_model, nhead, dropout)
        self.prot_to_rna = CrossModalCrossAttention(d_model, nhead, dropout)
        self.fusion_norm = nn.LayerNorm(d_model)
        # Improved gating with deeper network
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.LayerNorm(d_model * 4),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        # Modality-specific output projections
        self.rna_proj = nn.Linear(d_model, d_model)
        self.prot_proj = nn.Linear(d_model, d_model)

    def forward(self, rna_repr: Tensor, prot_repr: Tensor,
                rna_mask: Optional[Tensor] = None,
                prot_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Bidirectional cross-attention fusion.

        Returns:
            fused_rna:  RNA representation updated with protein context
            fused_prot: Protein representation updated with RNA context
        """
        # RNA -> Protein attention
        rna_updated = self.rna_to_prot(rna_repr, prot_repr, key_padding_mask=prot_mask)
        # Protein -> RNA attention
        prot_updated = self.prot_to_rna(prot_repr, rna_repr, key_padding_mask=rna_mask)

        # Gated fusion: combine original and cross-attended representations
        rna_gate = self.fusion_gate(torch.cat([rna_repr, rna_updated], dim=-1))
        prot_gate = self.fusion_gate(torch.cat([prot_repr, prot_updated], dim=-1))

        fused_rna = rna_repr + rna_gate * rna_updated
        fused_prot = prot_repr + prot_gate * prot_updated

        # Modality-specific projections with residual
        fused_rna = fused_rna + self.rna_proj(fused_rna)
        fused_prot = fused_prot + self.prot_proj(fused_prot)

        fused_rna = self.fusion_norm(fused_rna)
        fused_prot = self.fusion_norm(fused_prot)

        return fused_rna, fused_prot


# ---------------------------------------------------------------------------
#  Mechanism 3: Task-Adaptive Importance Masking
# ---------------------------------------------------------------------------

def task_adaptive_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.4,
    mask_value: int = -1,
    pad_value: int = 0,
    task_type: str = "perturbation",
    gene_importance: Optional[Union[torch.Tensor, np.ndarray]] = None,
    perturbed_genes_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """
    Dynamically adjust mask probabilities per gene based on the task type.
    Fully vectorized implementation (no Python per-row loops).
    Memory-optimized: operates directly on torch tensors when possible.

    Task-specific masking strategies:
      - "perturbation":       Higher mask ratio on known perturbation-target genes
      - "integration":        Balanced masking across batches; higher on HVGs
      - "large_perturbation": Higher on condition-specific differentially expressed
                              genes; global mask ratio lower due to dataset sparsity

    Args:
        values: (batch_size, seq_len) tokenized expression values.
        mask_ratio: Base mask ratio.
        mask_value: Value to fill at masked positions.
        pad_value: Value indicating padding.
        task_type: One of "perturbation", "integration", "large_perturbation".
        gene_importance: (seq_len,) Importance weights per gene column.
        perturbed_genes_mask: (seq_len,) Boolean mask indicating which genes are
                              known perturbation targets.

    Returns:
        masked_values: (batch_size, seq_len) with masked positions set to mask_value.
    """
    # Work directly on tensors when possible to avoid CPU round-trip
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    
    # Clone to avoid modifying original
    masked = values.clone()
    batch_size, seq_len = masked.shape
    device = masked.device

    # ---- Determine per-gene mask probabilities (vectorized) ----
    if task_type == "perturbation":
        base_prob = mask_ratio
        if perturbed_genes_mask is not None:
            if isinstance(perturbed_genes_mask, np.ndarray):
                perturbed_genes_mask = torch.from_numpy(perturbed_genes_mask)
            perturbed_genes_mask = perturbed_genes_mask.to(device, non_blocking=True)
            pert_boost = 0.3
            mask_len = min(len(perturbed_genes_mask), seq_len)
            gene_prob = torch.full((seq_len,), base_prob, dtype=torch.float, device=device)
            gene_prob[:mask_len] = torch.where(
                perturbed_genes_mask[:mask_len],
                torch.tensor(min(base_prob + pert_boost, 0.9), device=device),
                torch.tensor(base_prob, device=device),
            )
        else:
            gene_prob = torch.full((seq_len,), base_prob, device=device)
    elif task_type == "integration":
        base_prob = mask_ratio
        gene_prob = torch.full((seq_len,), base_prob, dtype=torch.float, device=device)
        if gene_importance is not None:
            if isinstance(gene_importance, np.ndarray):
                gene_importance = torch.from_numpy(gene_importance)
            gene_importance = gene_importance.to(device, non_blocking=True)
            imp_len = min(len(gene_importance), seq_len)
            imp = gene_importance[:imp_len]
            imp_min = imp.min()
            imp_max = imp.max()
            imp_norm = (imp - imp_min) / (imp_max - imp_min + 1e-8)
            gene_prob[:imp_len] = torch.clamp(base_prob + 0.2 * imp_norm, 0.1, 0.9)
    elif task_type == "large_perturbation":
        base_prob = mask_ratio
        gene_prob = torch.full((seq_len,), base_prob, dtype=torch.float, device=device)
        if perturbed_genes_mask is not None:
            if isinstance(perturbed_genes_mask, np.ndarray):
                perturbed_genes_mask = torch.from_numpy(perturbed_genes_mask)
            perturbed_genes_mask = perturbed_genes_mask.to(device, non_blocking=True)
            mask_len = min(len(perturbed_genes_mask), seq_len)
            gene_prob[:mask_len] = torch.where(
                perturbed_genes_mask[:mask_len],
                torch.tensor(min(base_prob + 0.25, 0.85), device=device),
                torch.tensor(base_prob, device=device),
            )
    else:
        gene_prob = torch.full((seq_len,), mask_ratio, device=device)

    # CLS token (position 0) is never masked
    gene_prob[0] = 0.0

    # ---- Fully vectorized masking using torch sampling ----
    # Generate random values for all positions at once
    rand_vals = torch.rand(batch_size, seq_len, device=device)
    # Bernoulli mask: each position masked independently with probability gene_prob
    pad_mask = masked == pad_value
    mask = (rand_vals < gene_prob.unsqueeze(0)) & ~pad_mask
    
    # Ensure at least 1 mask per row (for rows where no positions were masked)
    rows_no_mask = (~mask.any(dim=1)) & (~pad_mask.all(dim=1))
    if rows_no_mask.any():
        row_indices = torch.where(rows_no_mask)[0]
        for idx in row_indices:
            available = torch.where(~pad_mask[idx])[0]
            available = available[available != 0]  # exclude CLS
            if len(available) > 0:
                chosen = available[torch.randint(0, len(available), (1,))]
                mask[idx, chosen] = True
    
    # Apply mask
    masked[mask] = mask_value

    return masked.float()


# ---------------------------------------------------------------------------
#  PCT-AIM Transformer Model (main entry point)
# ---------------------------------------------------------------------------

class PCTAIMTransformerModel(nn.Module):
    """
    PCT-AIM Transformer: unified model for perturbation prediction, multi-omic
    integration, and large-scale perturbation tasks.

    Architecture highlights:
      - PerturbationConditionEncoder for explicit perturbation state awareness
      - CrossModalFusionLayer for RNA-Protein alignment
      - Task-adaptive masking integrated at the data-collation level
      - Gradient checkpointing support for memory-efficient training
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        use_grad_checkpointing: bool = False,
        # ---- PCT-AIM specific arguments ----
        use_pert_cond: bool = True,
        num_perturbations: Optional[int] = None,
        pert_per_gene: bool = True,
        use_cross_modal: bool = False,
        ntokens_mod: Optional[int] = None,
        vocab_mod: Optional[Any] = None,
        use_mod: bool = False,
        # ---- PCT-AIM: dedicated perturbation prediction ----
        use_pert_pred: bool = False,
        n_pert_pred_labels: Optional[int] = None,
    ):
        super().__init__()
        self.model_type = "PCTAIM-Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.use_pert_cond = use_pert_cond
        self.use_cross_modal = use_cross_modal
        self.use_mod = use_mod
        self.use_grad_checkpointing = use_grad_checkpointing

        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}")
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            if not flash_attn_available:
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning.")
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        # ---- Base encoders ----
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])

        # ---- Expression flag encoder (binary: 0=not expressed, 1=expressed) ----
        # This matches the pretrained scGPT whole-human model architecture.
        # The flag encoder helps the model distinguish zero vs non-zero expression.
        self.flag_encoder = nn.Embedding(2, d_model, padding_idx=None)
        nn.init.normal_(self.flag_encoder.weight, mean=0.0, std=0.02)

        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value)
        else:
            self.value_encoder = nn.Identity()

        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        # ---- [PCT-AIM] Perturbation Condition Encoder ----
        if use_pert_cond and num_perturbations is not None:
            self.pert_encoder = PerturbationConditionEncoder(
                num_perturbations, d_model, padding_idx=0,
                per_gene=pert_per_gene)

        # ---- [PCT-AIM] Perturbation Prediction Head ----
        self.use_pert_pred = use_pert_pred
        if use_pert_pred and n_pert_pred_labels is not None and n_pert_pred_labels > 0:
            self.pert_predictor = PerturbationPredictor(
                d_model, n_pert_pred_labels, nlayers=3)

        # ---- [PCT-AIM] Modality encoder for cross-modal ----
        if use_mod:
            self.mod_encoder = BatchLabelEncoder(
                ntokens_mod, d_model, padding_idx=vocab_mod[pad_token]
            )

        # ---- Domain-specific batchnorm ----
        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine)
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # ---- Main Transformer Encoder ----
        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout)
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model, nhead, d_hid, dropout,
                    batch_first=True, norm_scheme=self.norm_scheme)
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            # Use memory-efficient scaled_dot_product_attention if available (PyTorch 2.0+)
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout,
                batch_first=True, norm_first=pre_norm,
                bias=True)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # ---- [PCT-AIM] Cross-Modal Fusion Layer (after transformer) ----
        if use_cross_modal:
            self.cross_modal_fusion = CrossModalFusionLayer(d_model, nhead, dropout)

        # ---- Decoders ----
        decoder_kwargs = dict(
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
            use_mod=use_mod,
        )
        self.decoder = ExprDecoder(d_model, **decoder_kwargs)
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model, arch_style=mvc_decoder_style, **decoder_kwargs)

        if do_dab:
            self.grad_reverse_discriminator = EnhancedAdversarialDiscriminator(
                d_model, n_cls=num_batch_labels, reverse_grad=True, lambd=1.5)

        self.sim = Similarity(temp=0.1)
        self.creterion_cce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        # Initialize pert_encoder scale
        if hasattr(self, 'pert_encoder'):
            nn.init.constant_(self.pert_encoder.scale, 0.5)
            nn.init.constant_(self.pert_encoder.temperature, 0.5)
        # Xavier init for linear layers in value encoder
        if isinstance(self.value_encoder, ContinuousValueEncoder):
            for name, param in self.value_encoder.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
        # Initialize perturbation predictor final layer with small weights
        if hasattr(self, 'pert_predictor'):
            for name, param in self.pert_predictor.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Small init for stable start
                elif 'bias' in name:
                    nn.init.zeros_(param)
        # Initialize flag_encoder with better separation between expressed/non-expressed
        if hasattr(self, 'flag_encoder'):
            # Init not-expressed (index 0) with slightly negative bias
            # Init expressed (index 1) with slightly positive bias
            nn.init.normal_(self.flag_encoder.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Create a small separation between the two states
                if self.flag_encoder.weight.size(0) >= 2:
                    self.flag_encoder.weight[0] = -0.01 * torch.ones_like(self.flag_encoder.weight[0])
                    self.flag_encoder.weight[1] = 0.01 * torch.ones_like(self.flag_encoder.weight[1])
        # Initialize cross-modal gate to start with low cross-attention influence
        if self.use_cross_modal and hasattr(self, 'cross_modal_fusion'):
            for name, param in self.cross_modal_fusion.named_parameters():
                if 'gate_logit' in name:
                    nn.init.constant_(param, -2.0)  # Start with sigmoid gate ~0.12, low influence

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None,
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        # Gene token embeddings
        src = self.encoder(src)
        self.cur_gene_token_embs = src

        # Value embeddings
        values_emb = self.value_encoder(values)
        if self.input_emb_style == "scaling":
            values_emb = values_emb.unsqueeze(2)
            total_embs = src * values_emb
        else:
            total_embs = src + values_emb

        # ---- Expression flag embedding (binary: 0=not expressed, 1=expressed) ----
        # This matches the pretrained scGPT architecture and improves expression
        # modeling by explicitly encoding which genes are detected.
        flag = (values > 0).long()
        flag_emb = self.flag_encoder(flag)
        total_embs = total_embs + flag_emb

        # ---- [PCT-AIM] Fuse perturbation condition ----
        if self.use_pert_cond and hasattr(self, 'pert_encoder') and pert_labels is not None:
            pert_emb = self.pert_encoder(pert_labels)  # (batch, seq_len, d_model) or (batch, d_model)
            if pert_emb.dim() == 2:
                # Cell-level perturbation: expand to all tokens
                # shape (batch, d_model) -> (batch, 1, d_model) -> (batch, seq_len, d_model)
                pert_emb = pert_emb.unsqueeze(1).expand(-1, total_embs.size(1), -1)
            elif pert_emb.dim() == 3 and pert_emb.size(1) != total_embs.size(1):
                # Mismatch in seq_len: expand or slice
                if pert_emb.size(1) < total_embs.size(1):
                    # Repeat last element to fill
                    repeats = (total_embs.size(1) + pert_emb.size(1) - 1) // pert_emb.size(1)
                    pert_emb = pert_emb.repeat(1, repeats, 1)[:, :total_embs.size(1), :]
                else:
                    pert_emb = pert_emb[:, :total_embs.size(1), :]
            total_embs = total_embs + pert_emb

        # Batch label embedding (if needed for decoder later, we store separately)
        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(0, 2, 1)
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        # Main transformer (with optional gradient checkpointing)
        if self.use_grad_checkpointing and self.training:
            # NOTE: TransformerEncoder.forward(self, src, mask=None, src_key_padding_mask=None)
            # Pass None for mask (2nd pos arg) and src_key_padding_mask as 3rd pos arg.
            def _ckpt_forward(x, m, kpm):
                return self.transformer_encoder(x, mask=m, src_key_padding_mask=kpm)
            output = torch.utils.checkpoint.checkpoint(
                _ckpt_forward,
                total_embs,
                None,
                src_key_padding_mask,
                use_reentrant=False,
            )
        else:
            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask)
        return output  # (batch, seq_len, d_model)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True")

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
        mod_types: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None,
        # Cross-modal inputs (optional, only used when use_cross_modal=True)
        prot_src: Optional[Tensor] = None,
        prot_values: Optional[Tensor] = None,
        prot_key_padding_mask: Optional[Tensor] = None,
        # Perturbation prediction target
        pert_target: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass with support for:
          - pert_labels: perturbation condition labels
          - prot_src/prot_values: protein modality data for cross-modal attention
          - mod_types: modality type tokens (for multi-omic)
        """
        # ---- Encode RNA modality ----
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels, pert_labels)

        # ---- [PCT-AIM] Cross-Modal Fusion ----
        if self.use_cross_modal and prot_src is not None:
            # Encode protein modality through the same encoder (shared weights)
            prot_output = self._encode(
                prot_src, prot_values, prot_key_padding_mask, batch_labels, pert_labels)
            # Cross-modal fusion
            rna_fused, prot_fused = self.cross_modal_fusion(
                transformer_output, prot_output,
                rna_mask=src_key_padding_mask,
                prot_mask=prot_key_padding_mask)
            transformer_output = rna_fused  # use fused RNA representation

        # ---- Batch / modality embeddings for decoder conditioning ----
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)
        if self.use_mod:
            mod_emb = self.mod_encoder(mod_types)

        # Build decoder conditioning vector
        if self.use_batch_labels and self.use_mod:
            cat_0 = (batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
                     + mod_emb)
        elif self.use_batch_labels and not self.use_mod:
            cat_0 = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
        elif self.use_mod and not self.use_batch_labels:
            cat_0 = mod_emb
        else:
            cat_0 = None

        # ---- MLM Decoder ----
        output = {}
        mlm_output = self.decoder(
            transformer_output if cat_0 is None
            else torch.cat([transformer_output, cat_0], dim=2))
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        # ---- Cell embedding ----
        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb

        # ---- CLS ----
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)

        # ---- CCE (contrastive) - memory efficient - no second _encode call ----
        # Instead of calling _encode again (which doubles memory), we create
        # a perturbed view using dropout on the existing transformer output,
        # which preserves the contrastive signal while saving ~50% GPU memory.
        if CCE:
            cell1 = cell_emb
            # Use dropout-based perturbation on the transformer output to create
            # a second view. This is much more memory-efficient than running
            # a second _encode pass and works well for contrastive learning.
            if self.training:
                # Apply dropout as stochastic perturbation for contrastive view
                # v6: Increase dropout rate slightly for stronger augmentation
                drop_mask = torch.empty_like(transformer_output).bernoulli_(0.85)
                transformer_output2 = transformer_output * drop_mask
            else:
                transformer_output2 = transformer_output
            cell2 = self._get_cell_emb_from_layer(transformer_output2)
            if dist.is_initialized() and self.training:
                cls1_list = [torch.zeros_like(cell1) for _ in range(dist.get_world_size())]
                cls2_list = [torch.zeros_like(cell2) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2
                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # SimCLR-style NT-Xent loss with numerical stability
            # Normalize embeddings first for consistent cosine similarity
            cell1_norm = F.normalize(cell1, p=2, dim=1)
            cell2_norm = F.normalize(cell2, p=2, dim=1)
            # Compute similarity matrix: (batch, batch)
            cos_sim = torch.mm(cell1_norm, cell2_norm.t()) / self.sim.temp
            # Numerical stability: subtract max per row
            cos_sim_max, _ = cos_sim.max(dim=1, keepdim=True)
            cos_sim_stable = cos_sim - cos_sim_max.detach()
            # Compute log softmax manually for numerical stability
            exp_sim = torch.exp(cos_sim_stable)
            log_prob = cos_sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            loss_cce = -log_prob[range(len(labels)), labels].mean()
            output["loss_cce"] = loss_cce

        # ---- MVC ----
        if MVC:
            if self.use_batch_labels and self.use_mod:
                cat_1 = batch_emb + self._get_cell_emb_from_layer(mod_emb)
                cat_2 = (batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
                         + mod_emb)
            elif self.use_batch_labels and not self.use_mod:
                cat_1 = batch_emb
                cat_2 = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
            elif self.use_mod and not self.use_batch_labels:
                cat_1 = self._get_cell_emb_from_layer(mod_emb)
                cat_2 = mod_emb
            else:
                cat_1 = None
                cat_2 = None

            mvc_output = self.mvc_decoder(
                cell_emb if cat_1 is None else torch.cat([cell_emb, cat_1], dim=1),
                self.cur_gene_token_embs if cat_2 is None
                else torch.cat([self.cur_gene_token_embs, cat_2], dim=2))
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]

        # ---- ECS ----
        if ECS:
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = F.relu(cos_sim)
            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        # ---- [PCT-AIM] Perturbation Prediction ----
        if self.use_pert_pred and hasattr(self, 'pert_predictor'):
            pert_pred = self.pert_predictor(cell_emb)
            output["pert_pred"] = pert_pred
            if pert_target is not None:
                # Multi-label BCE loss
                output["loss_pert_pred"] = self.criterion_bce(pert_pred, pert_target.float())

        # ---- DAB (adversarial batch correction) ----
        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        N = src.size(0)
        device = next(self.parameters()).device
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = ((N, self.d_model) if time_step is not None
                 else (N, src.size(1), self.d_model))
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i:i + batch_size].to(device),
                values[i:i + batch_size].to(device),
                src_key_padding_mask[i:i + batch_size].to(device),
                batch_labels[i:i + batch_size].to(device) if batch_labels is not None else None,
                pert_labels[i:i + batch_size].to(device) if pert_labels is not None else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i:i + batch_size] = output
        return outputs

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None,
    ) -> Tensor:
        try:
            self._check_batch_labels(batch_labels)
        except Exception:
            warnings.warn("batch_labels required but not provided, using zeros")
            batch_labels = torch.zeros(cell_emb.shape[0], dtype=torch.long,
                                       device=cell_emb.device)

        src = self.encoder(src)
        if values is not None:
            values_emb = self.value_encoder(values)
            if self.input_emb_style == "scaling":
                values_emb = values_emb.unsqueeze(2)
                total_embs = src * values_emb
            else:
                total_embs = src + values_emb
        else:
            total_embs = src

        # Expression flag embedding
        if values is not None:
            flag = (values > 0).long()
            flag_emb = self.flag_encoder(flag)
            total_embs = total_embs + flag_emb

        if self.use_pert_cond and hasattr(self, 'pert_encoder') and pert_labels is not None:
            pert_emb = self.pert_encoder(pert_labels)
            if pert_emb.dim() == 2:
                pert_emb = pert_emb.unsqueeze(1).expand(-1, total_embs.size(1), -1)
            elif pert_emb.dim() == 3 and pert_emb.size(1) != total_embs.size(1):
                if pert_emb.size(1) < total_embs.size(1):
                    repeats = (total_embs.size(1) + pert_emb.size(1) - 1) // pert_emb.size(1)
                    pert_emb = pert_emb.repeat(1, repeats, 1)[:, :total_embs.size(1), :]
                else:
                    pert_emb = pert_emb[:, :total_embs.size(1), :]
            total_embs = total_embs + pert_emb

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(0, 2, 1)
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(total_embs.shape[:2], dtype=torch.bool,
                                               device=total_embs.device)
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask)

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)
        mlm_output = self.decoder(
            transformer_output if not self.use_batch_labels
            else torch.cat([transformer_output,
                            batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)], dim=2))
        return mlm_output["pred"]