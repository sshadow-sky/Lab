from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        ctx.coeff = coeff
        return input_tensor * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        max_iters: float = 1000.0,
        auto_step: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input_tensor, coeff)

    def step(self):
        self.iter_num += 1


class SubjectInvariantNorm(nn.Module):
    """
    Subject-wise normalization without subject-specific learnable identities.
    Input shape: [B, D], subject ids shape: [B].
    """

    def __init__(self, num_features: int, eps: float = 1e-5, min_count: int = 2):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.min_count = max(1, int(min_count))

        self.shared_gamma = nn.Parameter(torch.ones(1, num_features))
        self.shared_beta = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        bsz, feat_dim = x.shape
        if feat_dim != self.num_features:
            raise ValueError(f"SubjectInvariantNorm feature mismatch: got {feat_dim}, expected {self.num_features}")
        if subject_ids.ndim != 1 or subject_ids.shape[0] != bsz:
            raise ValueError("subject_ids must be shape [B]")

        out = torch.empty_like(x)
        unique_ids = torch.unique(subject_ids.detach().long())
        for sid_tensor in unique_ids:
            mask = subject_ids == sid_tensor
            idx = torch.where(mask)[0]
            chunk = x[idx]

            if chunk.shape[0] >= self.min_count:
                mean = chunk.mean(dim=0, keepdim=True)
                var = chunk.var(dim=0, unbiased=False, keepdim=True)
                out[idx] = (chunk - mean) / torch.sqrt(var + self.eps)
            else:
                out[idx] = F.layer_norm(x[idx], (feat_dim,), eps=self.eps)

        return out * self.shared_gamma + self.shared_beta


class FeatureBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(out_dim, min(max(in_dim, out_dim), out_dim * 2))
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return x


class CrossScaleAttention(nn.Module):
    def __init__(self, feature_dim: int, num_scales: int, num_heads: int, dropout: float):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.num_heads = self._valid_num_heads(feature_dim, num_heads)

        self.shared_attn = nn.MultiheadAttention(feature_dim, self.num_heads, dropout=dropout, batch_first=True)
        self.private_attn = nn.MultiheadAttention(feature_dim, self.num_heads, dropout=dropout, batch_first=True)
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

    def forward(self, shared_list: List[torch.Tensor], private_list: List[torch.Tensor]) -> List[torch.Tensor]:
        shared_tokens = torch.stack(shared_list, dim=1)  # [B, S, D]
        private_tokens = torch.stack(private_list, dim=1)  # [B, S, D]

        shared_ctx, _ = self.shared_attn(shared_tokens, shared_tokens, shared_tokens, need_weights=False)
        private_ctx, _ = self.private_attn(private_tokens, private_tokens, private_tokens, need_weights=False)
        shared_ctx = self.shared_norm(shared_tokens + shared_ctx)
        private_ctx = self.private_norm(private_tokens + private_ctx)

        alpha_s = torch.sigmoid(self.shared_gate(shared_ctx))
        private_clean = private_ctx.detach()
        alpha_p = torch.sigmoid(self.private_gate(private_clean))
        gamma = F.softplus(self.gamma_raw).view(1, self.num_scales, 1)
        clean_tokens = alpha_s * shared_ctx - gamma * alpha_p * private_clean
        return [clean_tokens[:, i, :] for i in range(self.num_scales)]


def diff_loss(diff: torch.Tensor, S: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * torch.mean(torch.sum(torch.sum(diff ** 2, dim=3) * S, dim=(1, 2)))


def F_norm_loss(S: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * torch.sum(torch.mean(S ** 2, dim=0))


class GraphLearn(nn.Module):
    """
    Learn graph adjacency from node differences.
    Input:  [N, V, F]
    Output: [N, V, V]
    """

    def __init__(self, alpha: float, num_of_features: int, device: torch.device):
        super().__init__()
        self.alpha = alpha
        self.a = nn.init.ones_(nn.Parameter(torch.empty(num_of_features, 1, device=device)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        n_graph, num_nodes, _ = x.shape
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # [N, V, V, F]
        tmpS = torch.exp(-F.relu(torch.matmul(torch.abs(diff), self.a).reshape(n_graph, num_nodes, num_nodes)))
        S = tmpS / (torch.sum(tmpS, dim=-1, keepdim=True) + 1e-6)
        ajloss = F_norm_loss(S, 1.0) + diff_loss(diff, S, self.alpha)
        return [S, ajloss]


class ChebConv(nn.Module):
    """
    K-order Chebyshev graph convolution after GraphLearn.
    Input:  [x:[N, V, F], W:[N, V, V]]
    Output: [N, V, H]
    """

    def __init__(self, num_of_filters: int, k: int, num_of_features: int, device: torch.device):
        super().__init__()
        self.Theta = nn.ParameterList(
            [nn.init.uniform_(nn.Parameter(torch.empty(num_of_features, num_of_filters, device=device))) for _ in range(k)]
        )
        self.out_channels = num_of_filters
        self.K = k
        self.device = device

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x, W = inputs
        n_graph, num_nodes, _ = x.shape

        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(n_graph, -1, -1)
        W = 0.5 * (W + W.transpose(1, 2)) + eye
        degree = torch.sum(W, dim=-1).clamp_min(1e-6)
        d_inv_sqrt = torch.pow(degree, -0.5)
        W_norm = d_inv_sqrt.unsqueeze(-1) * W * d_inv_sqrt.unsqueeze(1)
        L = eye - W_norm
        L_t = L - eye

        cheb_polynomials = [eye, L_t]
        for i in range(2, self.K):
            cheb_polynomials.append(2.0 * torch.matmul(L_t, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])

        output = torch.zeros(n_graph, num_nodes, self.out_channels, device=x.device)
        for k in range(self.K):
            rhs = torch.matmul(cheb_polynomials[k], x)
            output = output + torch.matmul(rhs, self.Theta[k])
        return F.relu(output)


class GCNBlock(nn.Module):
    def __init__(self, num_of_features: int, out_feature: int, alpha: float, k: int, device: torch.device):
        super().__init__()
        self.graph_learn = GraphLearn(alpha, num_of_features, device)
        self.cheb_conv = ChebConv(out_feature, k, num_of_features, device)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        S, ajloss = self.graph_learn(x)
        gcn = self.cheb_conv([x, S])
        return [gcn, S, ajloss]


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.norm(x)
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout1(x1)
        x2 = F.relu(self.fc2(x1))
        x2 = self.dropout2(x2)
        return [x1, x2]


class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc_layer1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc_layer2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.relu(self.fc_layer1(x))
        return self.fc_layer2(x)


class DMS_SGPAN(nn.Module):
    def __init__(self, net_params: Dict):
        super().__init__()
        self.device = net_params["DEVICE"]
        self.num_channels = int(net_params["num_of_vertices"])
        self.num_bands = int(net_params["num_of_features"])
        self.num_classes = int(net_params["category_number"])
        self.num_subjects = int(net_params["num_subjects"])

        self.graph_hidden = int(net_params.get("graph_hidden", 64))
        self.graph_readout_hidden = int(net_params.get("graph_readout_hidden", 256))
        self.gcl_readout_hidden = int(net_params.get("gcl_readout_hidden", 256))
        self.spectral_hidden = int(net_params.get("spectral_hidden", 128))
        self.dis_dim = int(net_params.get("disentangle_dim", 128))
        self.proj_dim = int(net_params.get("projection_dim", 64))
        self.dropout = float(net_params.get("dropout", 0.2))
        self.temperature = float(net_params.get("temperature", 0.2))
        self.ugfcda_warmup_epochs = max(0, int(net_params.get("ugfcda_warmup_epochs", 10)))
        self.ugfcda_eps = max(1e-12, float(net_params.get("ugfcda_eps", 1e-6)))
        self.ugfcda_reliability_threshold = max(
            0.0,
            min(1.0, float(net_params.get("ugfcda_reliability_threshold", 0.6))),
        )
        self.ugfcda_proto_align_weight = max(0.0, float(net_params.get("ugfcda_proto_align_weight", 0.1)))
        self.node_drop_rate = float(net_params.get("node_drop_rate", 0.15))
        self.edge_drop_rate = float(net_params.get("edge_drop_rate", 0.10))
        self.gcl_importance_protect = net_params.get("gcl_importance_protect", True)
        self.gcl_importance_centrality_weight = max(
            0.0,
            float(net_params.get("gcl_importance_centrality_weight", 0.5)),
        )
        self.gcl_importance_feature_weight = max(
            0.0,
            float(net_params.get("gcl_importance_feature_weight", 0.5)),
        )
        self.gcl_node_sample_temperature = max(
            1e-6,
            float(net_params.get("gcl_node_sample_temperature", 0.7)),
        )
        self.gcl_node_sample_eps = max(1e-12, float(net_params.get("gcl_node_sample_eps", 1e-6)))
        self.gcl_edge_protect_strength = max(
            0.0,
            min(1.0, float(net_params.get("gcl_edge_protect_strength", 0.7))),
        )
        self.gcl_edge_min_drop_scale = max(
            0.0,
            min(1.0, float(net_params.get("gcl_edge_min_drop_scale", 0.3))),
        )
        self.GLalpha = float(net_params.get("GLalpha", 0.01))
        self.cheb_k = int(net_params.get("K", 3))
        self.cross_scale_heads = int(net_params.get("cross_scale_heads", 4))
        self.frequency_band_groups = self._sanitize_frequency_band_groups(
            net_params.get("frequency_band_groups", None)
        )
        self.graph_band_indices = self._graph_frequency_band_indices()
        self.graph_num_bands = len(self.graph_band_indices)
        self.num_scales = len(self.frequency_band_groups)

        self.w_ce = float(net_params.get("w_ce", 1.0))
        self.w_gcl = float(net_params.get("w_gcl", 0.3))
        self.w_aj = float(net_params.get("w_aj", 0.2))
        self.w_align = float(net_params.get("w_align", 0.2))
        self.w_orth = float(net_params.get("w_orth", 0.5))
        self.w_subject = float(net_params.get("w_subject", 0.3))

        flat_dim = self.num_channels * self.num_bands
        self.gcl_keep_nodes = min(
            self.num_channels,
            max(1, int(round(self.num_channels * (1.0 - self.node_drop_rate)))),
        )
        self.ssbn = SubjectInvariantNorm(
            flat_dim,
            eps=float(net_params.get("ssbn_eps", 1e-5)),
            min_count=int(net_params.get("sin_min_count", 2)),
        )

        self.gcn = GCNBlock(
            num_of_features=self.graph_num_bands,
            out_feature=self.graph_hidden,
            alpha=self.GLalpha,
            k=self.cheb_k,
            device=self.device,
        )
        self.graph_readout = MLPBlock(
            self.num_channels * self.graph_hidden,
            self.graph_readout_hidden,
            self.graph_hidden,
            self.dropout,
        )
        self.gcl_readout = MLPBlock(
            self.gcl_keep_nodes * self.graph_hidden,
            self.gcl_readout_hidden,
            self.graph_hidden,
            self.dropout,
        )
        self.projector = Projector(self.graph_hidden, self.proj_dim)

        self.spectral_scale_encoders = nn.ModuleList(
            [
                FeatureBlock(self.num_channels * len(band_group), self.spectral_hidden, self.dropout)
                for band_group in self.frequency_band_groups
            ]
        )
        self.graph_feature_fuses = nn.ModuleList(
            [
                FeatureBlock(self.spectral_hidden + self.graph_hidden, self.spectral_hidden, self.dropout)
                for i in range(self.num_scales)
            ]
        )
        self.shared_encoders = nn.ModuleList(
            [
                FeatureBlock(self.spectral_hidden, self.dis_dim, self.dropout)
                for i in range(self.num_scales)
            ]
        )
        self.private_encoders = nn.ModuleList(
            [
                FeatureBlock(self.spectral_hidden, self.dis_dim, self.dropout)
                for i in range(self.num_scales)
            ]
        )
        self.cross_scale_attention = CrossScaleAttention(
            self.dis_dim,
            self.num_scales,
            num_heads=self.cross_scale_heads,
            dropout=self.dropout,
        )
        self.scale_gate = nn.Linear(self.dis_dim, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dis_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.dis_dim, self.num_classes),
        )

        self.subject_grl = GRL(
            alpha=1.0,
            lo=0.0,
            hi=1.0,
            max_iters=float(net_params.get("grl_max_iters", 2000)),
            auto_step=True,
        )
        subject_hidden = max(8, self.dis_dim // 2)
        self.shared_subject_discriminator = nn.Sequential(
            nn.LayerNorm(self.dis_dim),
            nn.Linear(self.dis_dim, subject_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(subject_hidden, self.num_subjects),
        )
        self.private_subject_classifier = nn.Sequential(
            nn.LayerNorm(self.dis_dim),
            nn.Linear(self.dis_dim, subject_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(subject_hidden, self.num_subjects),
        )

    def _default_frequency_band_groups(self) -> List[List[int]]:
        if self.num_bands >= 5:
            return [[0], [1], [2], [3], [4]]
        if self.num_bands >= 3:
            return [[i] for i in range(self.num_bands)][-3:]
        return [[i] for i in range(self.num_bands)]

    def _sanitize_frequency_band_groups(self, band_groups) -> List[List[int]]:
        if not isinstance(band_groups, (list, tuple)) or len(band_groups) == 0:
            return self._default_frequency_band_groups()

        valid_groups = []
        for group in band_groups:
            if isinstance(group, int):
                group = [group]
            if not isinstance(group, (list, tuple)):
                continue
            valid_group = sorted({int(idx) for idx in group if 0 <= int(idx) < self.num_bands})
            if valid_group:
                valid_groups.append(valid_group)
        return valid_groups if valid_groups else self._default_frequency_band_groups()

    def _graph_frequency_band_indices(self) -> List[int]:
        band_indices = sorted({idx for band_group in self.frequency_band_groups for idx in band_group})
        if band_indices:
            return band_indices
        return list(range(self.num_bands))

    def _ensure_shape(self, x: torch.Tensor) -> torch.Tensor:
        # DMS_SGPAN now treats each sample as one graph over EEG channels.
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, C, F], got shape={tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_bands:
            raise ValueError(
                f"Input feature mismatch: got C={x.shape[1]}, F={x.shape[2]}, expected C={self.num_channels}, F={self.num_bands}"
            )
        return x

    def _label_index(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim > 1:
            return y.argmax(dim=1)
        return y.long().view(-1)

    def _dynamic_graph(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, C, F]
        bsz = x.shape[0]
        graph_x = x[:, :, self.graph_band_indices]
        graph_feat, adj, ajloss = self.gcn(graph_x)

        graph_flat = graph_feat.reshape(bsz, -1)  # [B, C * H]
        _, graph_step_feat = self.graph_readout(graph_flat)  # [B, H]
        return [graph_feat, adj, graph_step_feat, ajloss]

    def _build_spectral_scales(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, C, F]. Each scale is a true frequency-band group over all channels.
        scale_features = []
        for band_group, encoder in zip(self.frequency_band_groups, self.spectral_scale_encoders):
            band_x = x[:, :, band_group].reshape(x.shape[0], -1)
            scale_features.append(encoder(band_x))
        return scale_features

    def _fuse_graph_features(self, graph_step_feat: torch.Tensor, feature_scales: List[torch.Tensor]) -> List[torch.Tensor]:
        # graph_step_feat: [B, H]
        fused = []
        for scale_idx, feature_scale in enumerate(feature_scales):
            fused.append(self.graph_feature_fuses[scale_idx](torch.cat([feature_scale, graph_step_feat], dim=-1)))
        return fused

    def _normalize_node_score(self, score: torch.Tensor) -> torch.Tensor:
        score_min = score.min(dim=-1, keepdim=True).values
        score_max = score.max(dim=-1, keepdim=True).values
        return (score - score_min) / (score_max - score_min + self.gcl_node_sample_eps)

    def _gcl_node_importance(self, graph_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_sym = 0.5 * (adj + adj.transpose(1, 2))
        centrality = self._normalize_node_score(adj_sym.sum(dim=-1))
        feature_energy = self._normalize_node_score(torch.norm(graph_feat, p=2, dim=-1))

        weight_sum = self.gcl_importance_centrality_weight + self.gcl_importance_feature_weight
        if weight_sum <= self.gcl_node_sample_eps:
            importance = 0.5 * (centrality + feature_energy)
        else:
            importance = (
                self.gcl_importance_centrality_weight * centrality
                + self.gcl_importance_feature_weight * feature_energy
            ) / weight_sum
        return importance.clamp(0.0, 1.0).detach()

    def _graph_aug_view(self, graph_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # graph_feat: [B, C, H], adj: [B, C, C]
        bsz, chn, hidden = graph_feat.shape
        keep_nodes = self.gcl_keep_nodes

        if self.gcl_importance_protect:
            node_importance = self._gcl_node_importance(graph_feat, adj)
            sample_prob = F.softmax(node_importance / self.gcl_node_sample_temperature, dim=-1)
            sample_prob = sample_prob.clamp_min(self.gcl_node_sample_eps)
            sample_prob = sample_prob / sample_prob.sum(dim=-1, keepdim=True)
            keep_idx = torch.multinomial(sample_prob, num_samples=keep_nodes, replacement=False)
            kept_importance = torch.gather(node_importance, dim=1, index=keep_idx)
        else:
            rand_score = torch.rand(bsz, chn, device=graph_feat.device)
            keep_idx = torch.topk(rand_score, k=keep_nodes, dim=-1).indices
            kept_importance = None

        feat_idx = keep_idx.unsqueeze(-1).expand(-1, -1, hidden)
        aug_feat = torch.gather(graph_feat, dim=1, index=feat_idx)  # [B, K, H]

        adj_row_idx = keep_idx.unsqueeze(-1).expand(-1, -1, chn)
        aug_adj = torch.gather(adj, dim=1, index=adj_row_idx)  # [B, K, C]
        adj_col_idx = keep_idx.unsqueeze(1).expand(-1, keep_nodes, -1)
        aug_adj = torch.gather(aug_adj, dim=2, index=adj_col_idx)  # [B, K, K]

        # Edge perturbation on the augmented graph: random drop + random weight jitter, then renormalize.
        if self.gcl_importance_protect and kept_importance is not None:
            edge_importance = torch.maximum(kept_importance.unsqueeze(1), kept_importance.unsqueeze(2))
            edge_drop_scale = (1.0 - self.gcl_edge_protect_strength * edge_importance).clamp(
                min=self.gcl_edge_min_drop_scale,
                max=1.0,
            )
            edge_drop_prob = self.edge_drop_rate * edge_drop_scale
        else:
            edge_drop_prob = self.edge_drop_rate
        edge_keep = (torch.rand_like(aug_adj) > edge_drop_prob).float()
        edge_scale = 1.0 + self.edge_drop_rate * (2.0 * torch.rand_like(aug_adj) - 1.0)
        aug_adj = aug_adj * edge_keep * edge_scale.clamp(min=0.0)

        eye = torch.eye(keep_nodes, device=graph_feat.device).view(1, keep_nodes, keep_nodes)
        aug_adj = aug_adj + eye * 1e-6
        aug_adj = aug_adj / (aug_adj.sum(dim=-1, keepdim=True) + 1e-6)

        aug_graph_feat = torch.einsum("bij,bjh->bih", aug_adj, aug_feat)
        aug_graph_feat = aug_graph_feat.reshape(bsz, -1)  # [B, K * H]
        _, aug_graph_feat = self.gcl_readout(aug_graph_feat)
        return aug_graph_feat  # [B, H]

    def _supervised_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z1, z2], dim=0)
        features = F.normalize(features, dim=-1)
        labels = labels.view(-1).repeat(2)

        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = torch.ones_like(logits)
        logits_mask.fill_diagonal_(0.0)

        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return torch.zeros((), device=features.device)
        loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
        return loss[valid].mean()

    def _source_supervised_graph_contrastive_loss(
        self,
        graph_feat: torch.Tensor,
        adj: torch.Tensor,
        graph_step_feat: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        aug_graph_feat = self._graph_aug_view(graph_feat, adj)
        z1 = self.projector(graph_step_feat)
        z2 = self.projector(aug_graph_feat)
        return self._supervised_contrastive_loss(z1, z2, source_label)

    def _encode_all(self, x: torch.Tensor, subject_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._ensure_shape(x)
        bsz = x.shape[0]

        flat = x.reshape(bsz, -1)
        flat_norm = self.ssbn(flat, subject_ids.long())
        x_norm = flat_norm.reshape(bsz, self.num_channels, self.num_bands)

        graph_feat, adj, graph_step_feat, ajloss = self._dynamic_graph(x_norm)

        feature_scales = self._build_spectral_scales(x_norm)
        fused_scales = self._fuse_graph_features(graph_step_feat, feature_scales)

        shared_seq_list = [self.shared_encoders[i](feat) for i, feat in enumerate(fused_scales)]
        private_seq_list = [self.private_encoders[i](feat) for i, feat in enumerate(fused_scales)]
        clean_shared_seq = self.cross_scale_attention(shared_seq_list, private_seq_list)

        clean_stack = torch.stack(clean_shared_seq, dim=1)  # [B, n, D]
        scale_weights = torch.softmax(self.scale_gate(clean_stack).squeeze(-1), dim=1)
        final_feat = torch.sum(scale_weights.unsqueeze(-1) * clean_stack, dim=1)
        logits = self.classifier(final_feat)

        return {
            "logits": logits,
            "final_feat": final_feat,
            "shared_scales": clean_shared_seq,
            "private_scales": private_seq_list,
            "shared_seq_scales": clean_shared_seq,
            "private_seq_scales": private_seq_list,
            "scale_weights": scale_weights,
            "graph_feat": graph_feat,
            "adj": adj,
            "graph_step_feat": graph_step_feat,
            "ajloss": ajloss,
        }

    def _subject_index(self, subject_ids: torch.Tensor) -> torch.Tensor:
        labels = subject_ids.long().view(-1)
        if labels.numel() == 0:
            return labels
        if labels.min() < 0 or labels.max() >= self.num_subjects:
            raise ValueError(
                f"subject_ids must be in [0, {self.num_subjects - 1}], "
                f"got min={int(labels.min().item())}, max={int(labels.max().item())}"
            )
        return labels

    def _subject_loss(
        self,
        shared_scales: List[torch.Tensor],
        private_scales: List[torch.Tensor],
        subject_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        labels = self._subject_index(subject_ids)
        num_scales = len(shared_scales)
        repeated_labels = labels.repeat(num_scales)

        shared_feat = torch.cat(shared_scales, dim=0)
        private_feat = torch.cat(private_scales, dim=0)

        shared_logits = self.shared_subject_discriminator(self.subject_grl(shared_feat))
        private_logits = self.private_subject_classifier(private_feat)
        shared_loss = F.cross_entropy(shared_logits, repeated_labels)
        private_loss = F.cross_entropy(private_logits, repeated_labels)
        subject_loss = 0.5 * (shared_loss + private_loss)
        shared_acc = (shared_logits.detach().argmax(dim=1) == repeated_labels).float().mean()
        private_acc = (private_logits.detach().argmax(dim=1) == repeated_labels).float().mean()
        return [subject_loss, shared_loss, private_loss, shared_acc, private_acc]

    def _ugfcda_build_batch_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.long().view(-1)
        feat_norm = F.normalize(features.detach(), dim=-1)
        feat_dim = feat_norm.shape[-1]
        prototypes = torch.zeros(self.num_classes, feat_dim, device=features.device)
        valid_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=features.device)

        for cls_id in range(self.num_classes):
            mask = labels == cls_id
            if mask.any():
                proto = feat_norm[mask].mean(dim=0)
                prototypes[cls_id] = F.normalize(proto, dim=0)
                valid_mask[cls_id] = True
        return prototypes, valid_mask

    def _ugfcda_empty_state(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "pseudo_labels": torch.zeros(0, dtype=torch.long, device=device),
            "reliability": torch.zeros(0, device=device),
            "feature_agreement": torch.zeros(0, device=device),
            "feature_margin": torch.zeros(0, device=device),
            "feature_entropy_score": torch.zeros(0, device=device),
            "scale_consistency": torch.zeros(0, device=device),
        }

    def _ugfcda_reliability_and_pseudo(
        self,
        source_scales: List[torch.Tensor],
        target_scales: List[torch.Tensor],
        source_label: torch.Tensor,
        target_scale_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source_labels = source_label.long().view(-1)
            if len(target_scales) == 0:
                return self._ugfcda_empty_state(source_label.device)

            device = target_scales[0].device
            source_proto_scales = []
            source_valid_scales = []
            for scale_idx, s_feat in enumerate(source_scales):
                proto, valid = self._ugfcda_build_batch_prototypes(s_feat, source_labels)
                source_proto_scales.append(proto)
                source_valid_scales.append(valid)
            source_proto_scales = torch.stack(source_proto_scales, dim=0)
            source_valid_scales = torch.stack(source_valid_scales, dim=0)

            if target_scale_weights is None or target_scale_weights.numel() == 0:
                scale_weights = torch.full(
                    (target_scales[0].shape[0], len(target_scales)),
                    1.0 / max(1, len(target_scales)),
                    device=device,
                )
            else:
                scale_weights = target_scale_weights.detach().float().to(device)
                scale_weights = scale_weights / (scale_weights.sum(dim=1, keepdim=True) + self.ugfcda_eps)

            feature_scores = []
            scale_predictions = []
            for scale_idx, proto in enumerate(source_proto_scales):
                valid_mask = source_valid_scales[scale_idx]
                t_feat = F.normalize(target_scales[scale_idx].detach(), dim=-1)
                scale_sim = torch.matmul(t_feat, proto.t()) / max(self.temperature, 1e-6)
                scale_sim = scale_sim.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                scale_score = torch.softmax(scale_sim, dim=1)
                feature_scores.append(scale_score)
                scale_predictions.append(scale_score.argmax(dim=1))

            feature_score_stack = torch.stack(feature_scores, dim=1)  # [B, S, C]
            feature_agreement_scores = torch.sum(feature_score_stack * scale_weights.unsqueeze(-1), dim=1)
            source_feature_valid = source_valid_scales.all(dim=0)
            feature_agreement_scores = feature_agreement_scores.masked_fill(~source_feature_valid.unsqueeze(0), 0.0)

            pseudo_labels = feature_agreement_scores.argmax(dim=1)
            feature_agreement = torch.gather(feature_agreement_scores, 1, pseudo_labels.view(-1, 1)).squeeze(1)
            top2 = torch.topk(feature_agreement_scores, k=min(2, self.num_classes), dim=1).values
            if top2.shape[1] > 1:
                feature_margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
            else:
                feature_margin = top2[:, 0].clamp(0.0, 1.0)

            score_dist = feature_agreement_scores / (feature_agreement_scores.sum(dim=1, keepdim=True) + self.ugfcda_eps)
            entropy = -(score_dist.clamp_min(self.ugfcda_eps) * score_dist.clamp_min(self.ugfcda_eps).log()).sum(dim=1)
            max_entropy = float(np.log(max(2, self.num_classes)))
            feature_entropy_score = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)

            scale_pred_stack = torch.stack(scale_predictions, dim=1)
            scale_consistency = (scale_pred_stack == pseudo_labels.unsqueeze(1)).float().mean(dim=1)

            reliability = (
                feature_agreement.pow(0.25)
                * feature_margin.pow(0.25)
                * feature_entropy_score.pow(0.25)
                * scale_consistency.pow(0.25)
            ).clamp(0.0, 1.0)

            return {
                "pseudo_labels": pseudo_labels,
                "reliability": reliability,
                "feature_agreement": feature_agreement,
                "feature_margin": feature_margin,
                "feature_entropy_score": feature_entropy_score,
                "scale_consistency": scale_consistency,
            }

    def _ugfcda_alignment_loss(
        self,
        source_scales: List[torch.Tensor],
        target_scales: List[torch.Tensor],
        source_label: torch.Tensor,
        target_pseudo_label: torch.Tensor,
        target_reliability: torch.Tensor,
        target_align_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_pseudo_label.numel() == 0:
            return torch.zeros((), device=source_scales[0].device)

        target_reliability = target_reliability.detach().float().view(-1).to(source_scales[0].device)
        keep = target_align_mask.detach().bool().view(-1).to(source_scales[0].device)
        if not keep.any():
            return torch.zeros((), device=source_scales[0].device)
        if target_reliability[keep].sum() <= self.ugfcda_eps:
            return torch.zeros((), device=source_scales[0].device)

        source_labels = source_label.long().view(-1)
        target_labels = target_pseudo_label.long().view(-1)
        source_proto_scales = []
        source_valid_scales = []
        for scale_idx, s_feat in enumerate(source_scales):
            proto, valid = self._ugfcda_build_batch_prototypes(s_feat, source_labels)
            source_proto_scales.append(proto)
            source_valid_scales.append(valid)
        source_proto_scales = torch.stack(source_proto_scales, dim=0)
        source_valid_scales = torch.stack(source_valid_scales, dim=0)

        t2s_losses = []
        proto_align = torch.zeros((), device=source_scales[0].device)
        for scale_idx, (s_feat, t_feat) in enumerate(zip(source_scales, target_scales)):
            source_prototypes = source_proto_scales[scale_idx]
            source_valid = source_valid_scales[scale_idx]
            feat_norm = F.normalize(t_feat[keep], dim=-1)
            logits = torch.matmul(feat_norm, source_prototypes.t()) / max(self.temperature, 1e-6)
            logits = logits.masked_fill(~source_valid.unsqueeze(0), -1e9)
            losses = F.cross_entropy(logits, target_labels[keep], reduction="none")
            t2s_losses.append(torch.sum(losses * target_reliability[keep]) / (target_reliability[keep].sum() + self.ugfcda_eps))

            target_proto = torch.zeros(self.num_classes, source_prototypes.shape[-1], device=source_prototypes.device)
            target_feat_norm = F.normalize(t_feat[keep], dim=-1)
            reliability_weight = target_reliability[keep]
            reliability_weight = reliability_weight / (reliability_weight.sum() + self.ugfcda_eps)
            for cls_id in range(self.num_classes):
                cls_mask = target_labels[keep] == cls_id
                if cls_mask.any():
                    cls_weight = reliability_weight[cls_mask].unsqueeze(-1)
                    target_proto[cls_id] = F.normalize(
                        torch.sum(target_feat_norm[cls_mask] * cls_weight, dim=0) / (cls_weight.sum() + self.ugfcda_eps),
                        dim=0,
                    )
            proto_dist = 1.0 - torch.sum(F.normalize(source_prototypes, dim=-1) * F.normalize(target_proto, dim=-1), dim=-1)
            target_valid = torch.bincount(target_labels[keep], minlength=self.num_classes).to(torch.bool)
            proto_valid = source_valid & target_valid
            if proto_valid.any():
                proto_align = proto_align + proto_dist[proto_valid].mean()

        t2s_loss = torch.stack(t2s_losses).mean() if t2s_losses else torch.zeros((), device=source_scales[0].device)
        total_align = t2s_loss + self.ugfcda_proto_align_weight * proto_align
        return total_align

    def _cross_covariance_loss(self, shared_scales: List[torch.Tensor], private_scales: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for shared, private in zip(shared_scales, private_scales):
            shared_centered = shared - shared.mean(dim=0, keepdim=True)
            private_centered = private - private.mean(dim=0, keepdim=True)
            shared_norm = shared_centered / (shared_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6)
            private_norm = private_centered / (private_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6)

            denom = max(1, shared.shape[0] - 1)
            cross_cov = torch.matmul(shared_norm.transpose(0, 1), private_norm) / float(denom)
            losses.append(cross_cov.pow(2).mean())
        return torch.stack(losses).mean()

    def forward(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        source_subject_ids: torch.Tensor,
        target_subject_ids: torch.Tensor,
        source_y: torch.Tensor,
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        cat_x = torch.cat([source_x, target_x], dim=0)
        cat_sid = torch.cat([source_subject_ids.long(), target_subject_ids.long()], dim=0)
        source_label = self._label_index(source_y)

        source_count = source_x.shape[0]
        target_count = target_x.shape[0]
        source_mask = torch.cat(
            [
                torch.ones(source_count, dtype=torch.bool, device=cat_x.device),
                torch.zeros(target_count, dtype=torch.bool, device=cat_x.device),
            ],
            dim=0,
        )
        source_label_all = torch.full(
            (source_count + target_count,),
            fill_value=-1,
            dtype=torch.long,
            device=cat_x.device,
        )
        source_label_all[:source_count] = source_label.to(cat_x.device)

        # Mix source/target before all encoders and keep the mixed order for every loss term.
        mixed_idx = torch.randperm(cat_x.shape[0], device=cat_x.device)
        cat_x = cat_x[mixed_idx]
        cat_sid = cat_sid[mixed_idx]
        source_mask = source_mask[mixed_idx]
        target_mask = ~source_mask
        source_label = source_label_all[mixed_idx][source_mask]

        enc = self._encode_all(cat_x, cat_sid)
        logits_s = enc["logits"][source_mask]
        logits_t = enc["logits"][target_mask]

        ce_loss = F.cross_entropy(logits_s, source_label)
        gcl_loss = self._source_supervised_graph_contrastive_loss(
            enc["graph_feat"][source_mask],
            enc["adj"][source_mask],
            enc["graph_step_feat"][source_mask],
            source_label,
        )

        target_prob = torch.softmax(logits_t.detach(), dim=1)
        target_confidence = target_prob.max(dim=1).values

        source_scales = [feat[source_mask] for feat in enc["shared_scales"]]
        target_scales = [feat[target_mask] for feat in enc["shared_scales"]]
        target_scale_weights = enc["scale_weights"][target_mask]
        ugfcda_state = self._ugfcda_reliability_and_pseudo(
            source_scales,
            target_scales,
            source_label,
            target_scale_weights,
        )
        target_pseudo = ugfcda_state["pseudo_labels"]
        target_reliability = ugfcda_state["reliability"]
        target_align_mask = (
            target_reliability >= self.ugfcda_reliability_threshold
            if int(current_epoch) >= self.ugfcda_warmup_epochs
            else torch.zeros_like(target_reliability, dtype=torch.bool)
        )
        align_active = bool(target_align_mask.any().item())
        if align_active:
            align_loss = self._ugfcda_alignment_loss(
                source_scales,
                target_scales,
                source_label,
                target_pseudo,
                target_reliability,
                target_align_mask,
            )
        else:
            align_loss = torch.zeros((), device=logits_s.device)
        subject_loss, subject_shared_loss, subject_private_loss, shared_subject_acc, private_subject_acc = self._subject_loss(
            enc["shared_scales"],
            enc["private_scales"],
            cat_sid,
        )
        orth_loss = self._cross_covariance_loss(enc["shared_scales"], enc["private_scales"])

        target_total = torch.tensor(float(max(1, target_count)), device=logits_s.device)
        target_align_count = target_align_mask.float().sum()
        target_align_coverage = target_align_count / target_total
        target_pseudo_class_counts = torch.bincount(
            target_pseudo.detach(),
            minlength=self.num_classes,
        ).float()
        target_align_class_counts = torch.bincount(
            target_pseudo[target_align_mask].detach(),
            minlength=self.num_classes,
        ).float()
        if target_align_mask.any():
            target_align_confidence = target_reliability[target_align_mask].mean()
        else:
            target_align_confidence = torch.zeros((), device=logits_s.device)

        target_feature_agreement = ugfcda_state["feature_agreement"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_feature_margin = ugfcda_state["feature_margin"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_feature_entropy_score = ugfcda_state["feature_entropy_score"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_scale_consistency = ugfcda_state["scale_consistency"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)

        total_loss = (
            self.w_ce * ce_loss
            + self.w_aj * enc["ajloss"]
            + self.w_gcl * gcl_loss
            + self.w_align * align_loss
            + self.w_orth * orth_loss
            + self.w_subject * subject_loss
        )

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "ajloss": enc["ajloss"],
            "gcl_loss": gcl_loss,
            "align_loss": align_loss,
            "orth_loss": orth_loss,
            "subject_loss": subject_loss,
            "subject_shared_loss": subject_shared_loss,
            "subject_private_loss": subject_private_loss,
            "shared_subject_acc": shared_subject_acc,
            "private_subject_acc": private_subject_acc,
            "target_pseudo_conf_mean": target_confidence.mean(),
            "target_reliability_mean": target_reliability.mean(),
            "target_feature_agreement_mean": target_feature_agreement,
            "target_feature_margin_mean": target_feature_margin,
            "target_feature_entropy_score_mean": target_feature_entropy_score,
            "target_scale_consistency_mean": target_scale_consistency,
            "target_align_conf_mean": target_align_confidence,
            "target_align_coverage": target_align_coverage,
            "target_align_count": target_align_count,
            "target_pseudo_class_counts": target_pseudo_class_counts,
            "target_align_class_counts": target_align_class_counts,
            "align_active": torch.tensor(float(align_active), device=logits_s.device),
            "source_logits": logits_s,
            "target_logits": logits_t,
            "source_labels": source_label,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, subject_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        if subject_ids is None:
            subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        enc = self._encode_all(x, subject_ids.long())
        return torch.softmax(enc["logits"], dim=1)
