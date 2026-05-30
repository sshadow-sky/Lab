import copy
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class TimeStepShuffle:
    @staticmethod
    def apply(source_data):
        if source_data.size(1) <= 1:
            return source_data
        source_data_1 = source_data.clone()
        cur_time_step = source_data_1[:, -1, :]
        dim_size = source_data[:, :-1, :].size(1)
        idxs = list(range(dim_size))
        random.shuffle(idxs)
        else_part = source_data_1[:, idxs, :]
        return torch.cat([else_part, cur_time_step.unsqueeze(1)], dim=1)


class MSE(nn.Module):
    def forward(self, pred, real):
        diffs = real - pred
        return torch.sum(diffs.pow(2)) / diffs.numel()


class SupervisedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.numel() == 0:
            return features.new_tensor(0.0)
        labels = labels.view(-1).long()
        if labels.numel() <= 1:
            return features.new_tensor(0.0)

        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.t()) / self.temperature

        logits_mask = torch.ones_like(logits, device=features.device)
        logits_mask.fill_(1.0)
        logits_mask = logits_mask - torch.eye(logits.size(0), device=features.device)

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

        positive_count = mask.sum(dim=1)
        valid = positive_count > 0
        if valid.sum() == 0:
            return features.new_tensor(0.0)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.proj = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))

    def forward(self, node_features):
        batch_size, node_count, _ = node_features.shape
        h = self.proj(node_features).view(batch_size, node_count, self.heads, self.out_features)
        src_score = (h * self.attn_src.view(1, 1, self.heads, self.out_features)).sum(dim=-1)
        dst_score = (h * self.attn_dst.view(1, 1, self.heads, self.out_features)).sum(dim=-1)
        score = self.leaky_relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1))
        alpha = torch.softmax(score, dim=2)
        alpha = self.dropout(alpha)
        out = torch.einsum("bijh,bjhd->bihd", alpha, h)
        out = out.reshape(batch_size, node_count, self.heads * self.out_features)
        return out


class SpatialFrequencyGATEncoder(nn.Module):
    def __init__(
        self,
        input_dim=310,
        num_channels=62,
        num_bands=5,
        gat_hidden_dim=16,
        gat_heads=4,
        model_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_time_steps=64,
    ):
        super().__init__()
        if num_channels * num_bands != input_dim:
            raise ValueError(
                f"input_dim={input_dim} must equal num_channels({num_channels}) * num_bands({num_bands})."
            )
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.max_time_steps = max_time_steps
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        self.model_dim = model_dim

        self.channel_gat = GraphAttentionLayer(num_bands, gat_hidden_dim, heads=gat_heads, dropout=dropout)
        self.channel_norm = nn.LayerNorm(gat_hidden_dim * gat_heads)
        self.channel_dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(num_channels * gat_hidden_dim * gat_heads, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_time_steps, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def _reshape_input(self, x):
        if x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        if x.dim() != 3:
            raise ValueError(f"Expected input with shape [B, T, F], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.size(-1)}")
        if x.size(1) > self.max_time_steps:
            raise ValueError(f"time_steps={x.size(1)} exceeds max_time_steps={self.max_time_steps}")
        return x.view(x.size(0), x.size(1), self.num_channels, self.num_bands)

    def encode_spatial_frequency(self, x):
        x = self._reshape_input(x)
        batch_size, time_steps, _, _ = x.shape
        channel_features = x.reshape(batch_size * time_steps, self.num_channels, self.num_bands)
        channel_out = self.channel_gat(channel_features)
        channel_out = self.channel_norm(channel_out)
        channel_out = self.channel_dropout(channel_out)
        fused = channel_out.reshape(batch_size, time_steps, -1)
        projected = self.input_projection(fused)
        projected = projected + self.positional_encoding[:, :time_steps, :]
        encoded = self.transformer(self.output_dropout(projected))
        encoded = self.output_norm(encoded)
        pooled = encoded.mean(dim=1)
        return encoded, pooled

    def forward(self, x):
        return self.encode_spatial_frequency(x)


class TransformerReconstructor(nn.Module):
    def __init__(
        self,
        model_dim=128,
        output_dim=310,
        transformer_heads=4,
        transformer_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_time_steps=64,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_time_steps = max_time_steps
        self.input_projection = nn.Linear(model_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_time_steps, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, latent_sequence):
        if latent_sequence.size(1) > self.max_time_steps:
            raise ValueError(
                f"time_steps={latent_sequence.size(1)} exceeds max_time_steps={self.max_time_steps}"
            )
        seq = self.input_projection(latent_sequence)
        seq = seq + self.positional_encoding[:, : latent_sequence.size(1), :]
        seq = self.transformer(self.dropout(seq))
        seq = self.output_norm(seq)
        return self.output_projection(seq)


class DMMRGATTransformerPreTrainingModel(nn.Module):
    def __init__(
        self,
        number_of_source=14,
        number_of_category=3,
        batch_size=10,
        time_steps=15,
        input_dim=310,
        num_channels=62,
        num_bands=5,
        gat_hidden_dim=16,
        gat_heads=4,
        model_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        decoder_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        temperature=0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.input_dim = input_dim
        self.sharedEncoder = SpatialFrequencyGATEncoder(
            input_dim=input_dim,
            num_channels=num_channels,
            num_bands=num_bands,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            model_dim=model_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_time_steps=max(time_steps, 64),
        )
        self.domainClassifier = nn.Linear(model_dim, number_of_source)
        self.decoders = nn.ModuleList(
            [
                TransformerReconstructor(
                    model_dim=model_dim,
                    output_dim=input_dim,
                    transformer_heads=transformer_heads,
                    transformer_layers=decoder_layers,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    max_time_steps=max(time_steps, 64),
                )
                for _ in range(number_of_source)
            ]
        )
        self.mse = MSE()
        self.contrastive = SupervisedInfoNCELoss(temperature=temperature)

    def forward(self, x, corres, subject_id, label_src, m=0.0, mark=0):
        del mark
        x = TimeStepShuffle.apply(x)
        shared_sequence, shared_last_out = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict, dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id.view(-1))
        contrast_loss = self.contrastive(shared_last_out, label_src.view(-1))

        corres = corres.view(corres.size(0), corres.size(1), -1)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)

        rec_loss = x.new_tensor(0.0)
        mix_subject_feature = None
        for decoder in self.decoders:
            x_out = decoder(shared_sequence)
            mix_subject_feature = x_out if mix_subject_feature is None else mix_subject_feature + x_out

        shared_sequence_2, shared_last_out_2 = self.sharedEncoder(mix_subject_feature)
        del shared_last_out_2
        for i, decoder in enumerate(self.decoders):
            x_out = decoder(shared_sequence_2)
            rec_loss = rec_loss + self.mse(x_out, splitted_tensors[i])

        return rec_loss, sim_loss, contrast_loss


class DMMRGATTransformerFineTuningModel(nn.Module):
    def __init__(
        self,
        base_model,
        number_of_source=14,
        number_of_category=3,
        batch_size=10,
        time_steps=15,
    ):
        super().__init__()
        self.baseModel = copy.deepcopy(base_model)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.sharedEncoder = self.baseModel.sharedEncoder
        model_dim = self.baseModel.sharedEncoder.model_dim
        self.cls_fc = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, number_of_category, bias=True),
        )

    def forward(self, x, label_src=0):
        shared_sequence, shared_last_out = self.sharedEncoder(x)
        del shared_sequence
        x_logits = self.cls_fc(shared_last_out)
        x_pred = F.log_softmax(x_logits, dim=1)
        cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        return x_pred, x_logits, cls_loss


class DMMRGATTransformerTestModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.baseModel = copy.deepcopy(base_model)

    def forward(self, x):
        shared_sequence, shared_last_out = self.baseModel.sharedEncoder(x)
        del shared_sequence
        x_shared_logits = self.baseModel.cls_fc(shared_last_out)
        return x_shared_logits


def build_correspondence_batch(source_batches, source_labels, reference_labels):
    label_data_dict_list = []
    for data_one_subject, label_one_subject in zip(source_batches, source_labels):
        cur_map = defaultdict(list)
        for one_data, one_label in zip(data_one_subject, label_one_subject):
            cur_map[int(one_label)].append(one_data)
        label_data_dict_list.append(cur_map)

    corres_batch_data = []
    for one_map in label_data_dict_list:
        for one_label in reference_labels:
            label_cur = int(one_label)
            candidate = one_map[label_cur]
            if len(candidate) == 0:
                raise RuntimeError(f"No correspondence sample found for label {label_cur} in a source subject batch.")
            corres_batch_data.append(random.choice(candidate))
    return torch.stack(corres_batch_data)
