import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans


param_path = "config/model_param/PCLTDGCN.yaml"


def _load_pcl_param():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
        return cfg.get("params", {}), cfg.get("train", {})
    except IOError:
        print("\n{} may not exist or not available".format(param_path))
        return {}, {}


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.se(self.maxpool(x))
        avg_out = self.se(self.avgpool(x))
        return self.sigmoid(max_out + avg_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        return self.sigmoid(output)


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        ca = self.ca(x)
        out = x * ca
        sa = self.sa(out)
        out = out * sa
        return out + residual, ca, sa


class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        h = x
        for _ in range(self.diffusion_step):
            if adj.dim() == 3:
                h = torch.einsum("bcnt,bnm->bcmt", h, adj).contiguous()
            else:
                h = torch.einsum("bcnt,nm->bcmt", h, adj).contiguous()
            out.append(h)

        h = torch.cat(out, dim=1)
        h = self.conv(h)
        return self.dropout(h)


class Graph_Generator(nn.Module):
    def __init__(self, channels=5, num_nodes=62, topk_ratio=0.8):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)
        self.topk_ratio = topk_ratio

    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous() / math.sqrt(x.shape[1])
            ),
            -1,
        )

        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous() / math.sqrt(x.shape[1])
            ),
            -1,
        )

        adj_f = torch.cat([adj_dyn_1.unsqueeze(-1), adj_dyn_2.unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(-1), -1)

        k = max(int(adj_f.shape[1] * self.topk_ratio), 1)
        _, topk_indices = torch.topk(adj_f, k=k, dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        return adj_f * mask


class DGCN(nn.Module):
    def __init__(self, channels=5, num_nodes=62, diffusion_step=1, dropout=0.1, topk_ratio=0.8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator = Graph_Generator(channels, num_nodes, topk_ratio)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        return x + skip, adj_dyn


class MHGCN(nn.Module):
    def __init__(self, layers, chan_num, band_num, diffusion_step=1, dropout=0.1, topk_ratio=0.8):
        super().__init__()
        self.HGCN_layers = nn.ModuleList()
        for _ in range(layers):
            self.HGCN_layers.append(
                DGCN(channels=band_num, num_nodes=chan_num, diffusion_step=diffusion_step, dropout=dropout, topk_ratio=topk_ratio)
            )

    def forward(self, x):
        output = [x]
        adjs = []
        h = x
        for layer in self.HGCN_layers:
            h, adj = layer(h)
            output.append(h)
            adjs.append(adj)
        out = torch.cat(output, dim=1)
        return out, adjs


class Encoder(nn.Module):
    def __init__(
        self,
        in_planes=(5, 62),
        layers=2,
        hidden_2=64,
        diffusion_step=1,
        dropout=0.25,
        cbam_reduction=4,
        cbam_kernel_size=3,
        topk_ratio=0.8,
    ):
        super().__init__()
        self.chan_num = in_planes[1]
        self.band_num = in_planes[0]

        self.GGCN = MHGCN(
            layers=layers,
            chan_num=self.chan_num,
            band_num=self.band_num,
            diffusion_step=diffusion_step,
            dropout=dropout,
            topk_ratio=topk_ratio,
        )
        self.CBAM = CBAMBlock(channel=(layers + 1) * self.band_num, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
        self.fc1 = nn.Linear(self.chan_num * (layers + 1) * self.band_num, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        # Accept inputs with shape either:
        #  - (B, band, chan)
        #  - (B, chan, band)
        #  - (B, band, chan, feat) or (B, chan, band, feat) where the last dim is an extra feature/channel (e.g. multiple band features)
        # If there is an extra trailing feature dim, collapse it by mean so the tensor becomes 3D.
        if x.dim() == 4:
            # collapse last dim if present (e.g. (B, sample_length, chan, band_feat) -> (B, sample_length, chan))
            if x.size(-1) > 1:
                x = x.mean(dim=-1)
            else:
                x = x.squeeze(-1)

        if x.dim() != 3:
            raise ValueError(f"Unexpected input dimensionality {tuple(x.shape)}, expected 3D or 4D tensor")

        # now x is (B, D1, D2) where D1/D2 correspond to (band,chan) in some order
        if x.size(1) == self.chan_num and x.size(2) == self.band_num:
            x = x.permute(0, 2, 1).contiguous()
        elif x.size(1) != self.band_num or x.size(2) != self.chan_num:
            raise ValueError(
                "Unexpected input shape {}, expected [B, {}, {}] or [B, {}, {}]".format(
                    tuple(x.shape), self.band_num, self.chan_num, self.chan_num, self.band_num
                )
            )
        x = x.unsqueeze(3)
        g_feat, g_adj = self.GGCN(x)
        g_feat, ca, sa = self.CBAM(g_feat)

        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        return out, [g_adj, ca, sa]


class ClassClassifier(nn.Module):
    def __init__(self, hidden_2, num_cls):
        super().__init__()
        self.classifier = nn.Linear(hidden_2, num_cls)

    def forward(self, x):
        return self.classifier(x)


class DomainAdaptationModel(nn.Module):
    def __init__(
        self,
        in_planes=(5, 62),
        layers=2,
        hidden_1=256,
        hidden_2=64,
        num_of_class=3,
        device="cuda:0",
        source_num=1,
        target_num=1,
    ):
        super().__init__()

        model_param, _ = _load_pcl_param()
        self.layers = int(model_param.get("layers", layers))
        self.hidden_1 = int(model_param.get("hidden_1", hidden_1))
        self.hidden_2 = int(model_param.get("hidden_2", hidden_2))
        self.diffusion_step = int(model_param.get("diffusion_step", 1))
        self.dropout = float(model_param.get("dropout", 0.25))
        self.cbam_reduction = int(model_param.get("cbam_reduction", 4))
        self.cbam_kernel_size = int(model_param.get("cbam_kernel_size", 3))
        self.topk_ratio = float(model_param.get("topk_ratio", 0.8))
        self.target_topk_ratio = float(model_param.get("target_topk_ratio", 0.3))
        self.tem = float(model_param.get("temperature", 1.0))
        self.ema_factor = float(model_param.get("ema_factor", 0.8))

        self.encoder = Encoder(
            in_planes=in_planes,
            layers=self.layers,
            hidden_2=self.hidden_2,
            diffusion_step=self.diffusion_step,
            dropout=self.dropout,
            cbam_reduction=self.cbam_reduction,
            cbam_kernel_size=self.cbam_kernel_size,
            topk_ratio=self.topk_ratio,
        )
        self.cls_classifier = ClassClassifier(hidden_2=self.hidden_2, num_cls=num_of_class)

        self.source_f_bank = torch.zeros(source_num, self.hidden_2)
        self.target_f_bank = torch.zeros(target_num, self.hidden_2)
        self.source_score_bank = torch.zeros(source_num, num_of_class).to(device)
        self.target_score_bank = torch.zeros(target_num, num_of_class).to(device)

        self.num_of_class = num_of_class
        self.device = device

    def forward(self, source, target, source_label, source_index, target_index, current_epoch, max_epochs):
        source_f, [self.src_adj, self.src_sa, self.src_ca] = self.encoder(source)
        target_f, [self.tar_adj, self.tar_sa, self.tar_ca] = self.encoder(target)

        source_predict = self.cls_classifier(source_f)
        target_predict = self.cls_classifier(target_f)

        source_label_feature = F.softmax(source_predict, dim=1)
        target_label_feature = F.softmax(target_predict, dim=1)

        src_sim, src_prototype = self._get_source_similar(source_f, source_label_feature, source_index)
        tgt_sim, tgt_prototype, tat_cluster_label = self._get_target_similar(
            target_f, target_label_feature, target_index, src_prototype
        )

        s2t_pro = self._get_st_similar(source_f, tgt_prototype)
        t2s_pro = self._get_st_similar(target_f, src_prototype)
        s2s_pro = self._get_st_similar(source_f, src_prototype)
        t2t_pro = self._get_st_similar(target_f, tgt_prototype)

        return (
            source_predict,
            source_f,
            target_predict,
            target_f,
            [self.src_adj, self.src_sa, self.src_ca],
            [self.tar_adj, self.tar_sa, self.tar_ca],
            src_sim,
            tgt_sim,
            tat_cluster_label,
            s2t_pro,
            t2s_pro,
            s2s_pro,
            t2t_pro,
        )

    def _get_source_similar(self, feature_source_f, source_label_feature, source_index):
        self.eval()
        output_f = F.normalize(feature_source_f, p=2, dim=1)

        self.source_f_bank[source_index] = output_f.detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        prototype_class = []
        pred_labels = torch.argmax(self.source_score_bank, dim=1).cpu()
        for class_id in range(self.num_of_class):
            source_feature = self.source_f_bank[pred_labels == class_id]
            if source_feature.size(0) > 0:
                prototype = source_feature.mean(dim=0)
            else:
                prototype = torch.zeros(output_f.size(1))
            prototype_class.append(prototype)

        prototypes = torch.stack(prototype_class)
        src_sim = torch.mm(output_f.to(self.device), F.normalize(prototypes.to(self.device), p=2, dim=1).T) / self.tem
        return src_sim, prototypes

    def _get_target_similar(self, feature_target_f, target_label_feature, target_index, src_prototype):
        self.eval()
        f = F.normalize(feature_target_f, p=2, dim=1)

        self.target_f_bank[target_index] = f.detach().clone().cpu()
        self.target_score_bank[target_index] = target_label_feature.detach().clone()

        output = self.target_f_bank.to(self.device)
        scores = self.target_score_bank
        aggregated_scores = scores.max(dim=1)[0]
        num_samples = len(aggregated_scores)

        k = max(int(num_samples * self.target_topk_ratio), 1)
        _, top_indices = torch.topk(aggregated_scores, k)
        output_f = output[top_indices]

        n_clusters = min(self.num_of_class, max(output_f.shape[0], 1))
        if n_clusters < self.num_of_class:
            pad = torch.zeros((self.num_of_class - n_clusters, output_f.shape[1]), device=self.device)
            prototype = torch.cat([output_f[:n_clusters], pad], dim=0)
        else:
            kmeans = KMeans(n_clusters=self.num_of_class, random_state=0)
            kmeans.fit(output_f.cpu().detach().numpy())
            prototype = torch.tensor(kmeans.cluster_centers_, device=self.device)

        tgt_sim = torch.mm(F.normalize(f, p=2, dim=1), F.normalize(prototype, p=2, dim=1).T) / self.tem
        target_predict = F.softmax(tgt_sim, dim=1)
        tar_label = torch.argmax(target_predict, dim=1)
        return tgt_sim, prototype, tar_label

    def _get_st_similar(self, feature, prototypes):
        if prototypes.numel() == 0:
            return torch.zeros((feature.size(0), self.num_of_class), device=feature.device)

        feature = F.normalize(feature, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        st_sim = torch.mm(feature.to(self.device), prototypes.to(self.device).T) / self.tem
        return F.softmax(st_sim, dim=1)

    def get_init_banks(self, source, source_index):
        self.eval()
        with torch.no_grad():
            source_f, _ = self.encoder(source)
            source_predict = self.cls_classifier(source_f)
            source_label_feature = F.softmax(source_predict, dim=1)
            self.source_f_bank[source_index] = F.normalize(source_f, p=2, dim=1).detach().clone().cpu()
            self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def get_init_banks_tgt(self, tgt, tgt_index):
        self.eval()
        with torch.no_grad():
            tgt_f, _ = self.encoder(tgt)
            tgt_predict = self.cls_classifier(tgt_f)
            tgt_label_feature = F.softmax(tgt_predict, dim=1)
            self.target_f_bank[tgt_index] = F.normalize(tgt_f, p=2, dim=1).detach().clone().cpu()
            self.target_score_bank[tgt_index] = tgt_label_feature.detach().clone()

    def target_predict(self, feature_target):
        self.eval()
        with torch.no_grad():
            target_f, _ = self.encoder(feature_target)
            target_predict = self.cls_classifier(target_f)
            return F.softmax(target_predict, dim=1)


class PCLTDGCN(nn.Module):
    """
    Classification wrapper kept for compatibility with Model dict usage.
    For full PCL-TDGCN training, use DomainAdaptationModel in PCLTDGCN_train.py.
    """

    def __init__(self, num_electrodes, feature_dim, num_classes):
        super().__init__()
        self.model = DomainAdaptationModel(
            in_planes=(int(feature_dim), int(num_electrodes)),
            num_of_class=int(num_classes),
            source_num=1,
            target_num=1,
            device="cpu",
        )

    def forward(self, x):
        f, _ = self.model.encoder(x)
        return self.model.cls_classifier(f)
