"""
DBHPLNet-SP: Dual-Branch Hyperbolic Prototype Learning Network
with Shared-Private Disentanglement.

This file extends the reduced DBHPLNet core by adding back:

1. shared encoders
2. private encoders
3. cross-scale attention
4. orthogonality loss between shared/private features
5. subject discrimination loss:
   - shared features -> GRL -> subject discriminator
   - private features -> subject classifier

It reuses the core modules from dbhplnet_core.py:
  - FeatureBlock
  - ChannelGraphBranch
  - HyperbolicPrototypeHead

Forward:
  out = model(
      source_x,
      source_y,
      target_x=target_x,
      source_subject_ids=source_subject_ids,
      target_subject_ids=target_subject_ids,
      epoch=epoch,
  )

Input:
  source_x / target_x: [B, C, F]

If subject ids are not passed, subject_loss is set to zero while orth_loss
remains active.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    from .dbhplnet_core import (
        FeatureBlock,
        ChannelGraphBranch,
        HyperbolicPrototypeHead,
    )
except ImportError:
    from dbhplnet_core import (
        FeatureBlock,
        ChannelGraphBranch,
        HyperbolicPrototypeHead,
    )


# ---------------------------------------------------------------------------
# Gradient reversal layer
# ---------------------------------------------------------------------------

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


class GRL(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        max_iters: float = 2000.0,
        auto_step: bool = True,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_iters = float(max_iters)
        self.auto_step = bool(auto_step)
        self.iter_num = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = (
            2.0 * (self.hi - self.lo)
            / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )
        if self.auto_step:
            self.iter_num += 1
        return GradientReverseFunction.apply(x, float(coeff))


# ---------------------------------------------------------------------------
# Frequency scale branch
# ---------------------------------------------------------------------------

class FrequencyScaleBranch(nn.Module):
    """
    Frequency branch that returns one token per frequency-band group.

    Unlike the previous simplified FrequencyBandBranch that aggregated all
    band groups immediately, this branch keeps scale tokens so they can be
    separately processed by shared/private encoders.
    """

    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        hidden_dim: int = 128,
        band_groups: Optional[List[List[int]]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.band_groups = self._sanitize_groups(band_groups)
        self.num_scales = len(self.band_groups)

        self.group_encoders = nn.ModuleList(
            [
                FeatureBlock(
                    in_dim=self.num_channels * len(group),
                    out_dim=hidden_dim,
                    dropout=dropout,
                )
                for group in self.band_groups
            ]
        )

    def _default_groups(self) -> List[List[int]]:
        if self.num_bands >= 5:
            return [[1, 2], [3], [4]]
        return [[i] for i in range(self.num_bands)]

    def _sanitize_groups(self, groups) -> List[List[int]]:
        if not groups:
            return self._default_groups()
        valid = []
        for group in groups:
            if isinstance(group, int):
                group = [group]
            group = sorted({int(i) for i in group if 0 <= int(i) < self.num_bands})
            if group:
                valid.append(group)
        return valid if valid else self._default_groups()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        tokens = []
        for group, encoder in zip(self.band_groups, self.group_encoders):
            group_x = x[:, :, group].reshape(x.shape[0], -1)
            tokens.append(encoder(group_x))
        return tokens


# ---------------------------------------------------------------------------
# Shared/private feature refinement
# ---------------------------------------------------------------------------

class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for shared/private features.

    shared/private lists:
      list of [B, D]

    output:
      clean shared list, same length and shape.
    """

    def __init__(self, feature_dim: int, num_scales: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_scales = int(num_scales)
        self.num_heads = self._valid_num_heads(feature_dim, num_heads)

        self.shared_attn = nn.MultiheadAttention(
            feature_dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.private_attn = nn.MultiheadAttention(
            feature_dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.shared_norm = nn.LayerNorm(feature_dim)
        self.private_norm = nn.LayerNorm(feature_dim)

        self.shared_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.private_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.gamma_raw = nn.Parameter(torch.full((num_scales,), -2.0))

    @staticmethod
    def _valid_num_heads(feature_dim: int, requested_heads: int) -> int:
        requested_heads = max(1, int(requested_heads))
        for heads in range(requested_heads, 0, -1):
            if feature_dim % heads == 0:
                return heads
        return 1

    def forward(
        self,
        shared_list: List[torch.Tensor],
        private_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        shared_tokens = torch.stack(shared_list, dim=1)    # [B, S, D]
        private_tokens = torch.stack(private_list, dim=1)  # [B, S, D]

        shared_ctx, _ = self.shared_attn(
            shared_tokens,
            shared_tokens,
            shared_tokens,
            need_weights=False,
        )
        private_ctx, _ = self.private_attn(
            private_tokens,
            private_tokens,
            private_tokens,
            need_weights=False,
        )

        shared_ctx = self.shared_norm(shared_tokens + shared_ctx)
        private_ctx = self.private_norm(private_tokens + private_ctx)

        alpha_s = torch.sigmoid(self.shared_gate(shared_ctx))
        alpha_p = torch.sigmoid(self.private_gate(private_ctx.detach()))
        gamma = F.softplus(self.gamma_raw).view(1, self.num_scales, 1)

        clean_tokens = alpha_s * shared_ctx - gamma * alpha_p * private_ctx.detach()
        return [clean_tokens[:, i, :] for i in range(self.num_scales)]


class SharedPrivateDualBranchEncoder(nn.Module):
    """
    Dual-branch backbone + shared/private encoders.

    Pipeline:
      Input [B, C, F]
        -> frequency scale tokens
        -> channel graph feature
        -> per-scale fusion
        -> shared/private encoders
        -> cross-scale attention
        -> final shared feature z
    """

    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        spectral_hidden: int = 128,
        graph_hidden: int = 64,
        dis_dim: int = 128,
        dropout: float = 0.2,
        band_groups: Optional[List[List[int]]] = None,
        graph_bands: Optional[List[int]] = None,
        cross_scale_heads: int = 4,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)

        self.band_branch = FrequencyScaleBranch(
            num_channels=num_channels,
            num_bands=num_bands,
            hidden_dim=spectral_hidden,
            band_groups=band_groups,
            dropout=dropout,
        )
        self.graph_branch = ChannelGraphBranch(
            num_channels=num_channels,
            num_bands=num_bands,
            graph_hidden=graph_hidden,
            graph_bands=graph_bands,
            dropout=dropout,
        )

        self.num_scales = self.band_branch.num_scales

        self.scale_fusers = nn.ModuleList(
            [
                FeatureBlock(
                    in_dim=spectral_hidden + graph_hidden,
                    out_dim=spectral_hidden,
                    dropout=dropout,
                )
                for _ in range(self.num_scales)
            ]
        )
        self.shared_encoders = nn.ModuleList(
            [FeatureBlock(spectral_hidden, dis_dim, dropout) for _ in range(self.num_scales)]
        )
        self.private_encoders = nn.ModuleList(
            [FeatureBlock(spectral_hidden, dis_dim, dropout) for _ in range(self.num_scales)]
        )

        self.cross_scale_attention = CrossScaleAttention(
            feature_dim=dis_dim,
            num_scales=self.num_scales,
            num_heads=cross_scale_heads,
            dropout=dropout,
        )
        self.scale_gate = nn.Linear(dis_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B, C, F], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_bands:
            raise ValueError(
                f"Expected C={self.num_channels}, F={self.num_bands}; "
                f"got C={x.shape[1]}, F={x.shape[2]}"
            )

        band_tokens = self.band_branch(x)
        graph_feature, adj, adj_loss = self.graph_branch(x)

        fused_scales = []
        for idx, token in enumerate(band_tokens):
            fused_scales.append(
                self.scale_fusers[idx](
                    torch.cat([token, graph_feature], dim=-1)
                )
            )

        shared_raw = [
            encoder(feat)
            for encoder, feat in zip(self.shared_encoders, fused_scales)
        ]
        private = [
            encoder(feat)
            for encoder, feat in zip(self.private_encoders, fused_scales)
        ]
        shared_clean = self.cross_scale_attention(shared_raw, private)

        shared_stack = torch.stack(shared_clean, dim=1)
        scale_weights = torch.softmax(self.scale_gate(shared_stack).squeeze(-1), dim=1)
        final_feature = torch.sum(shared_stack * scale_weights.unsqueeze(-1), dim=1)

        return {
            "feature": final_feature,
            "shared_scales": shared_clean,
            "private_scales": private,
            "shared_raw_scales": shared_raw,
            "fused_scales": fused_scales,
            "graph_feature": graph_feature,
            "band_tokens": band_tokens,
            "scale_weights": scale_weights,
            "adj": adj,
            "adj_loss": adj_loss,
        }


def cross_covariance_orth_loss(
    shared_scales: List[torch.Tensor],
    private_scales: List[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize cross-covariance between shared and private features."""
    losses = []
    for shared, private in zip(shared_scales, private_scales):
        shared_centered = shared - shared.mean(dim=0, keepdim=True)
        private_centered = private - private.mean(dim=0, keepdim=True)

        shared_norm = shared_centered / (
            shared_centered.std(dim=0, unbiased=False, keepdim=True) + eps
        )
        private_norm = private_centered / (
            private_centered.std(dim=0, unbiased=False, keepdim=True) + eps
        )

        denom = max(1, shared.shape[0] - 1)
        cross_cov = torch.matmul(shared_norm.t(), private_norm) / float(denom)
        losses.append(cross_cov.pow(2).mean())
    return torch.stack(losses).mean()


class SubjectDisentanglementLoss(nn.Module):
    """
    Subject-related disentanglement loss.

    shared features:
      GRL -> subject discriminator
      Minimize CE with GRL, so the feature extractor learns subject-invariant
      shared features.

    private features:
      subject classifier without GRL
      Minimize CE, so private features preserve subject information.
    """

    def __init__(
        self,
        feature_dim: int,
        num_subjects: int,
        dropout: float = 0.2,
        grl_max_iters: int = 2000,
    ):
        super().__init__()
        self.num_subjects = int(num_subjects)
        self.grl = GRL(max_iters=grl_max_iters, auto_step=True)

        hidden = max(8, int(feature_dim) // 2)
        self.shared_subject_discriminator = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_subjects),
        )
        self.private_subject_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_subjects),
        )

    def forward(
        self,
        shared_scales: List[torch.Tensor],
        private_scales: List[torch.Tensor],
        subject_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = subject_ids.long().view(-1)
        if labels.numel() == 0:
            raise ValueError("subject_ids is empty.")
        if labels.min() < 0 or labels.max() >= self.num_subjects:
            raise ValueError(
                f"subject_ids must be in [0, {self.num_subjects - 1}], "
                f"got min={int(labels.min().item())}, max={int(labels.max().item())}"
            )

        num_scales = len(shared_scales)
        repeated_labels = labels.repeat(num_scales)

        shared_feat = torch.cat(shared_scales, dim=0)
        private_feat = torch.cat(private_scales, dim=0)

        shared_logits = self.shared_subject_discriminator(self.grl(shared_feat))
        private_logits = self.private_subject_classifier(private_feat)

        shared_loss = F.cross_entropy(shared_logits, repeated_labels)
        private_loss = F.cross_entropy(private_logits, repeated_labels)
        subject_loss = 0.5 * (shared_loss + private_loss)

        shared_acc = (
            shared_logits.detach().argmax(dim=1) == repeated_labels
        ).float().mean()
        private_acc = (
            private_logits.detach().argmax(dim=1) == repeated_labels
        ).float().mean()

        return {
            "subject_loss": subject_loss,
            "subject_shared_loss": shared_loss,
            "subject_private_loss": private_loss,
            "shared_subject_acc": shared_acc,
            "private_subject_acc": private_acc,
        }


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

class DBHPLNetSP(nn.Module):
    """
    DBHPLNet-SP:
    Dual-Branch Hyperbolic Prototype Learning Network with Shared-Private
    Disentanglement.
    """

    def __init__(self, net_params: Dict):
        super().__init__()

        self.num_channels = int(net_params["num_of_vertices"])
        self.num_bands = int(net_params["num_of_features"])
        self.num_classes = int(net_params["category_number"])
        self.num_subjects = int(net_params.get("num_subjects", 1))

        self.dis_dim = int(
            net_params.get("disentangle_dim", net_params.get("feature_dim", 128))
        )

        self.w_ce = float(net_params.get("w_ce", 1.0))
        self.w_aj = float(net_params.get("w_aj", 0.2))
        self.w_align = float(net_params.get("w_align", 0.2))
        self.w_orth = float(net_params.get("w_orth", 0.5))
        self.w_subject = float(net_params.get("w_subject", 0.3))
        self.w_proto = float(net_params.get("ugfcda_proto_align_weight", 0.1))
        self.warmup_epochs = int(net_params.get("ugfcda_warmup_epochs", 10))

        dropout = float(net_params.get("dropout", 0.2))

        self.encoder = SharedPrivateDualBranchEncoder(
            num_channels=self.num_channels,
            num_bands=self.num_bands,
            spectral_hidden=int(net_params.get("spectral_hidden", 128)),
            graph_hidden=int(net_params.get("graph_hidden", 64)),
            dis_dim=self.dis_dim,
            dropout=dropout,
            band_groups=net_params.get("frequency_band_groups", None),
            graph_bands=net_params.get("graph_band_indices", None),
            cross_scale_heads=int(net_params.get("cross_scale_heads", 4)),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dis_dim),
            nn.Dropout(dropout),
            nn.Linear(self.dis_dim, self.num_classes),
        )

        self.subject_loss_module = SubjectDisentanglementLoss(
            feature_dim=self.dis_dim,
            num_subjects=self.num_subjects,
            dropout=dropout,
            grl_max_iters=int(net_params.get("grl_max_iters", 2000)),
        )

        self.proto_head = HyperbolicPrototypeHead(
            num_classes=self.num_classes,
            temperature=float(net_params.get("temperature", 0.2)),
            curvature=float(net_params.get("hyperbolic_curvature", 1.0)),
            tangent_scale=float(net_params.get("hyperbolic_tangent_scale", 0.5)),
            reliability_threshold=float(net_params.get("ugfcda_reliability_threshold", 0.6)),
            eps=float(net_params.get("ugfcda_eps", 1e-6)),
            ball_eps=float(net_params.get("hyperbolic_ball_eps", 1e-5)),
        )

    def _label_index(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim > 1:
            return y.argmax(dim=1)
        return y.long().view(-1)

    def _zero(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def encode(self, x: torch.Tensor) -> Dict:
        enc = self.encoder(x)
        enc["logits"] = self.classifier(enc["feature"])
        return enc

    def forward(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: Optional[torch.Tensor] = None,
        source_subject_ids: Optional[torch.Tensor] = None,
        target_subject_ids: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        source_label = self._label_index(source_y)

        source_enc = self.encode(source_x)
        logits_s = source_enc["logits"]
        ce_loss = F.cross_entropy(logits_s, source_label)

        target_enc = None
        logits_t = None
        pseudo_state = None
        align_loss = self._zero(source_x.device)

        if target_x is not None:
            target_enc = self.encode(target_x)
            logits_t = target_enc["logits"]

            with torch.no_grad():
                source_proto, source_valid = self.proto_head.build_prototypes(
                    source_enc["feature"].detach(),
                    source_label.detach(),
                )
                pseudo_state = self.proto_head.pseudo_label(
                    target_enc["feature"].detach(),
                    source_proto,
                    source_valid,
                )

            if int(epoch) >= self.warmup_epochs:
                align_loss = self.proto_head.alignment_loss(
                    source_features=source_enc["feature"],
                    source_labels=source_label,
                    target_features=target_enc["feature"],
                    target_pseudo=pseudo_state["pseudo"],
                    target_reliability=pseudo_state["reliability"],
                    target_keep=pseudo_state["keep"],
                    proto_align_weight=self.w_proto,
                )

        # Orthogonality loss is always active.
        if target_enc is not None:
            orth_shared = [
                torch.cat([s_feat, t_feat], dim=0)
                for s_feat, t_feat in zip(
                    source_enc["shared_scales"],
                    target_enc["shared_scales"],
                )
            ]
            orth_private = [
                torch.cat([s_feat, t_feat], dim=0)
                for s_feat, t_feat in zip(
                    source_enc["private_scales"],
                    target_enc["private_scales"],
                )
            ]
        else:
            orth_shared = source_enc["shared_scales"]
            orth_private = source_enc["private_scales"]

        orth_loss = cross_covariance_orth_loss(orth_shared, orth_private)

        # Subject loss is active only when subject ids are provided.
        subject_outputs = {
            "subject_loss": self._zero(source_x.device),
            "subject_shared_loss": self._zero(source_x.device),
            "subject_private_loss": self._zero(source_x.device),
            "shared_subject_acc": self._zero(source_x.device),
            "private_subject_acc": self._zero(source_x.device),
        }

        if source_subject_ids is not None:
            if target_enc is not None:
                if target_subject_ids is None:
                    raise ValueError(
                        "target_subject_ids must be provided when target_x is provided "
                        "and subject loss is enabled."
                    )
                subject_ids = torch.cat(
                    [source_subject_ids.long(), target_subject_ids.long()],
                    dim=0,
                ).to(source_x.device)
                subject_shared = orth_shared
                subject_private = orth_private
            else:
                subject_ids = source_subject_ids.long().to(source_x.device)
                subject_shared = source_enc["shared_scales"]
                subject_private = source_enc["private_scales"]

            subject_outputs = self.subject_loss_module(
                subject_shared,
                subject_private,
                subject_ids,
            )

        total_loss = (
            self.w_ce * ce_loss
            + self.w_aj * source_enc["adj_loss"]
            + self.w_align * align_loss
            + self.w_orth * orth_loss
            + self.w_subject * subject_outputs["subject_loss"]
        )

        out = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "adj_loss": source_enc["adj_loss"],
            "align_loss": align_loss,
            "orth_loss": orth_loss,
            "subject_loss": subject_outputs["subject_loss"],
            "subject_shared_loss": subject_outputs["subject_shared_loss"],
            "subject_private_loss": subject_outputs["subject_private_loss"],
            "shared_subject_acc": subject_outputs["shared_subject_acc"],
            "private_subject_acc": subject_outputs["private_subject_acc"],
            "logits_s": logits_s,
            "logits_t": logits_t,
            "source_feature": source_enc["feature"],
            "source_shared_scales": source_enc["shared_scales"],
            "source_private_scales": source_enc["private_scales"],
            "source_scale_weights": source_enc["scale_weights"],
            "source_adj": source_enc["adj"],
        }

        if pseudo_state is not None:
            out.update(
                {
                    "target_pseudo": pseudo_state["pseudo"],
                    "target_reliability": pseudo_state["reliability"],
                    "target_align_mask": pseudo_state["keep"],
                }
            )

        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.encode(x)["logits"], dim=1)


# Alias options.
DBHPLNet = DBHPLNetSP
HyperProtoSharedPrivateNet = DBHPLNetSP
