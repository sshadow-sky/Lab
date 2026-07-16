from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden_dim = max(out_dim, min(max(in_dim, out_dim), out_dim * 2))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelGraphBranch(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        graph_hidden: int = 64,
        graph_bands: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        if graph_bands is None:
            graph_bands = list(range(self.num_bands))
        self.graph_bands = [int(i) for i in graph_bands if 0 <= int(i) < self.num_bands]
        if not self.graph_bands:
            self.graph_bands = list(range(self.num_bands))

        self.node_encoder = FeatureBlock(len(self.graph_bands), graph_hidden, dropout)
        self.readout = FeatureBlock(self.num_channels * graph_hidden, graph_hidden, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_x = x[:, :, self.graph_bands]
        node_h = self.node_encoder(node_x.reshape(-1, len(self.graph_bands)))
        node_h = node_h.reshape(x.shape[0], self.num_channels, -1)

        node_norm = F.normalize(node_h, p=2, dim=-1)
        adj = torch.matmul(node_norm, node_norm.transpose(1, 2))
        adj = torch.softmax(adj, dim=-1)

        graph_h = torch.matmul(adj, node_h)
        graph_feature = self.readout(graph_h.reshape(x.shape[0], -1))
        eye = torch.eye(self.num_channels, device=x.device).unsqueeze(0)
        adj_loss = (adj - eye).pow(2).mean()
        return graph_feature, adj, adj_loss


class PoincareBall:
    """Poincare ball operations with constant negative curvature -c.

    The encoder/backbone feature is still an ordinary Euclidean vector.
    This class is only used inside the prototype branch:
        Euclidean feature -> tangent vector at origin -> expmap_0 -> Poincare ball.
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-6, ball_eps: float = 1e-5):
        self.c = max(float(curvature), float(eps))
        self.eps = float(eps)
        self.ball_eps = float(ball_eps)

    @property
    def sqrt_c(self) -> float:
        return math.sqrt(self.c)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points into the open Poincare ball."""
        max_norm = (1.0 - self.ball_eps) / self.sqrt_c
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space at the origin to the ball."""
        tangent_norm = torch.norm(tangent, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        mapped = torch.tanh(self.sqrt_c * tangent_norm) * tangent / (self.sqrt_c * tangent_norm)
        return self.project(mapped)

    def logmap0(self, point: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from the ball to tangent space at the origin."""
        point = self.project(point)
        point_norm = torch.norm(point, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        scaled_norm = (self.sqrt_c * point_norm).clamp(
            min=0.0,
            max=1.0 - self.ball_eps,
        )
        artanh = 0.5 * (torch.log1p(scaled_norm) - torch.log1p(-scaled_norm))
        return artanh * point / (self.sqrt_c * point_norm)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise geodesic distance on the Poincare ball.

        Args:
            x: [N, D]
            y: [M, D]

        Returns:
            distance matrix [N, M]
        """
        x = self.project(x)
        y = self.project(y)

        diff_sq = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
        x_sq = x.pow(2).sum(dim=-1, keepdim=True)
        y_sq = y.pow(2).sum(dim=-1).unsqueeze(0)

        denom = (
            (1.0 - self.c * x_sq).clamp_min(self.ball_eps)
            * (1.0 - self.c * y_sq).clamp_min(self.ball_eps)
        )
        acosh_arg = 1.0 + 2.0 * self.c * diff_sq / denom
        acosh_arg = acosh_arg.clamp_min(1.0 + self.eps)
        return torch.acosh(acosh_arg) / self.sqrt_c


class HyperbolicPrototypeHead(nn.Module):
    """Hyperbolic prototype head on a Poincare ball.

    Kept public API:
        build_prototypes(features, labels)
        pseudo_label(target_features, source_proto, source_valid)
        alignment_loss(...)

    Therefore DBHPL.py and DBHPL_train.py do not need to change.

    Internal replacement:
        old:
            prototypes = Euclidean mean
            pseudo label = torch.cdist in Euclidean space
            alignment = squared Euclidean distance

        new:
            features -> normalize -> tangent scaling -> expmap_0
            prototypes = logmap_0 -> weighted mean -> expmap_0
            pseudo label = negative squared Poincare distance
            alignment = reliable target CE + matched hyperbolic prototype distance
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.2,
        curvature: float = 1.0,
        tangent_scale: float = 0.5,
        reliability_threshold: float = 0.6,
        eps: float = 1e-6,
        ball_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = max(float(temperature), float(eps))
        self.reliability_threshold = float(reliability_threshold)
        self.eps = float(eps)
        self.curvature = max(float(curvature), float(eps))
        self.tangent_scale = max(float(tangent_scale), float(eps))
        self.ball_eps = float(ball_eps)
        self.ball = PoincareBall(
            curvature=self.curvature,
            eps=self.eps,
            ball_eps=self.ball_eps,
        )

    def _to_ball(self, features: torch.Tensor) -> torch.Tensor:
        """Map Euclidean encoder features to the Poincare ball."""
        tangent = F.normalize(features, dim=-1) * self.tangent_scale
        return self.ball.expmap0(tangent)

    def _weighted_hyperbolic_mean(
        self,
        features: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Approximate Frechet mean using tangent-space averaging.

        points on ball -> logmap_0 -> weighted Euclidean mean -> expmap_0.
        """
        points = self._to_ball(features)
        if weights is None:
            weights = torch.ones(
                points.shape[0],
                device=points.device,
                dtype=points.dtype,
            )
        weights = weights.float().view(-1)
        weights = weights / (weights.sum() + self.eps)

        tangent = self.ball.logmap0(points)
        tangent_mean = torch.sum(tangent * weights.unsqueeze(-1), dim=0)
        return self.ball.expmap0(tangent_mean.unsqueeze(0)).squeeze(0)

    def build_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """Build class prototypes on the Poincare ball.

        Args:
            features: Euclidean encoder features [N, D]
            labels: integer labels [N]
            weights: optional reliability weights [N]

        Returns:
            prototypes: hyperbolic prototypes on the ball [C, D]
            valid: whether each class has at least one sample [C]
        """
        feat_dim = features.shape[-1]
        prototypes = torch.zeros(
            self.num_classes,
            feat_dim,
            device=features.device,
            dtype=features.dtype,
        )
        valid = torch.zeros(self.num_classes, device=features.device, dtype=torch.bool)

        labels = labels.long().view(-1)
        for cls in range(self.num_classes):
            mask = labels == cls
            if mask.any():
                cls_weights = weights[mask] if weights is not None else None
                prototypes[cls] = self._weighted_hyperbolic_mean(
                    features[mask],
                    cls_weights,
                )
                valid[cls] = True
        return prototypes, valid

    def _logits_to_prototypes(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Class logits from negative squared Poincare distance."""
        points = self._to_ball(features)
        dist = self.ball.distance(points, prototypes)
        return -dist.pow(2) / self.temperature

    @torch.no_grad()
    def pseudo_label(self, target_features: torch.Tensor, source_proto: torch.Tensor, source_valid: torch.Tensor):
        """Generate target pseudo labels using hyperbolic prototype distance."""
        if target_features.numel() == 0 or not source_valid.any():
            count = target_features.shape[0]
            return {
                "pseudo": torch.zeros(count, dtype=torch.long, device=target_features.device),
                "reliability": torch.zeros(count, device=target_features.device),
                "keep": torch.zeros(count, dtype=torch.bool, device=target_features.device),
            }

        logits = self._logits_to_prototypes(target_features, source_proto)
        logits = logits.masked_fill(~source_valid.unsqueeze(0), -1e9)
        prob = torch.softmax(logits, dim=1)

        pseudo = prob.argmax(dim=1)
        confidence = prob.gather(1, pseudo.view(-1, 1)).squeeze(1)

        top2 = torch.topk(prob, k=min(2, self.num_classes), dim=1).values
        if top2.shape[1] > 1:
            margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
        else:
            margin = top2[:, 0].clamp(0.0, 1.0)

        # Stable reliability score.  This is stricter than confidence alone
        # but not as easy to collapse as direct multiplication.
        reliability = torch.sqrt(
            confidence.clamp_min(0.0) * margin.clamp_min(0.0)
        )
        keep = reliability >= self.reliability_threshold

        return {
            "pseudo": pseudo,
            "reliability": reliability,
            "keep": keep,
            "confidence": confidence,
            "margin": margin,
        }

    def alignment_loss(
        self,
        source_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor,
        target_pseudo: torch.Tensor,
        target_reliability: torch.Tensor,
        target_keep: torch.Tensor,
        proto_align_weight: float = 0.1,
    ) -> torch.Tensor:
        """Reliability-weighted source-target hyperbolic prototype alignment.

        Two terms are used:
          1. reliable target CE against source hyperbolic prototypes
          2. matched source/target prototype geodesic distance
        """
        source_proto, source_valid = self.build_prototypes(source_features, source_labels)

        if target_keep.numel() == 0 or not target_keep.any():
            return torch.zeros((), device=source_features.device)

        kept_features = target_features[target_keep]
        kept_labels = target_pseudo[target_keep].long()
        kept_reliability = target_reliability[target_keep].detach().float()

        if kept_reliability.sum() <= self.eps:
            return torch.zeros((), device=source_features.device)

        logits = self._logits_to_prototypes(kept_features, source_proto)
        logits = logits.masked_fill(~source_valid.unsqueeze(0), -1e9)

        target_ce = F.cross_entropy(logits, kept_labels, reduction="none")
        target_ce = (
            target_ce * kept_reliability
        ).sum() / (kept_reliability.sum() + self.eps)

        target_proto, target_valid = self.build_prototypes(
            kept_features,
            kept_labels,
            kept_reliability,
        )
        proto_valid = source_valid & target_valid

        if proto_valid.any():
            proto_distance = self.ball.distance(source_proto, target_proto).diagonal()
            proto_loss = proto_distance[proto_valid].mean()
        else:
            proto_loss = torch.zeros((), device=source_features.device)

        return target_ce + float(proto_align_weight) * proto_loss
