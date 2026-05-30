import math
import random

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


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 1.0, max_iters: float = 1000.0, auto_step: bool = False):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        coeff = float(2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters)) - (self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input_tensor, coeff)

    def step(self):
        self.iter_num += 1


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


class Discriminator(nn.Module):
    def __init__(self, hidden_1: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class Discriminator3(nn.Module):
    def __init__(self, hidden_1: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class DomainAdversarialLoss(nn.Module):
    def __init__(self, hidden_1: int, reduction: str = "mean", max_iter: int = 100):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones(len(d_s), 1, device=x.device)
        d_label_t = torch.zeros(len(d_t), 1, device=x.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class DomainAdversarialLossThreeAda(nn.Module):
    def __init__(self, hidden_1: int, reduction: str = "mean", max_iter: int = 100):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)
        source_num = int(len(x) / 3)
        d_s = d[0 : 2 * source_num, :]
        d_t = d[2 * source_num :, :]
        d_label_s = torch.ones(2 * source_num, 1, device=x.device)
        d_label_t = torch.zeros(source_num, 1, device=x.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class TripleDomainAdversarialLoss(nn.Module):
    def __init__(self, hidden_1: int, max_iter: int = 100):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator3(hidden_1)
        self.domain_discriminator_accuracy = None
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)
        source_num = int(len(x) / 3)
        d_label_s = torch.ones((source_num, 1), device=x.device)
        d_label_t = torch.zeros((source_num, 1), device=x.device)
        d_label_u = torch.ones((source_num, 1), device=x.device) + 1
        label = torch.cat((d_label_s, d_label_t, d_label_u)).squeeze().long()
        return self.criterion(d, label)


class SelfAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(attention_weights), values)


def transpose_qkv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


def diff_loss(diff: torch.Tensor, s_matrix: torch.Tensor, f_alpha: float) -> torch.Tensor:
    if len(s_matrix.shape) == 4:
        return f_alpha * torch.mean(torch.sum(torch.sum(diff ** 2, axis=3) * s_matrix, axis=(1, 2)))
    return f_alpha * torch.sum(torch.matmul(s_matrix, torch.sum(diff ** 2, axis=2)))


def f_norm_loss(s_matrix: torch.Tensor, f_alpha: float) -> torch.Tensor:
    if len(s_matrix.shape) == 3:
        return f_alpha * torch.sum(torch.mean(s_matrix ** 2, axis=0))
    return f_alpha * torch.sum(s_matrix ** 2)


class GraphLearn(nn.Module):
    def __init__(self, alpha: float, num_of_features: int, device: torch.device):
        super().__init__()
        self.alpha = alpha
        self.a = nn.init.ones_(nn.Parameter(torch.FloatTensor(num_of_features, 1).to(device)))

    def forward(self, x: torch.Tensor):
        n, v, f = x.shape
        diff = (x.expand(v, n, v, f).permute(2, 1, 0, 3) - x.expand(v, n, v, f)).permute(1, 0, 2, 3)
        tmp_s = torch.exp(-F.relu(torch.reshape(torch.matmul(torch.abs(diff), self.a), [n, v, v])))
        s_matrix = tmp_s / torch.sum(tmp_s, axis=1, keepdims=True)
        s_loss = f_norm_loss(s_matrix, 1)
        d_loss = diff_loss(diff, s_matrix, self.alpha)
        adj_loss = s_loss + d_loss
        return s_matrix, adj_loss


class ChebConv(nn.Module):
    def __init__(self, num_of_filters: int, k: int, num_of_features: int, device: torch.device):
        super().__init__()
        self.theta = nn.ParameterList(
            [nn.init.uniform_(nn.Parameter(torch.FloatTensor(num_of_features, num_of_filters).to(device))) for _ in range(k)]
        )
        self.out_channels = num_of_filters
        self.k = k
        self.device = device

    def forward(self, inputs):
        x, w = inputs
        n, v, _ = x.shape
        d = torch.diag_embed(torch.sum(w, axis=1))
        lap = d - w
        lambda_max = 2.0
        l_t = (2 * lap) / lambda_max - torch.eye(int(v), device=self.device)
        cheb_polynomials = [torch.eye(int(v), device=self.device), l_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * l_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

        graph_signal = x
        output = torch.zeros(n, v, self.out_channels, device=self.device)
        for k_idx in range(self.k):
            t_k = cheb_polynomials[k_idx]
            theta_k = self.theta[k_idx]
            rhs = t_k.matmul(graph_signal)
            output = output + rhs.matmul(theta_k)
        return F.relu(output)


class GCNBlock(nn.Module):
    def __init__(self, net_params: dict):
        super().__init__()
        self.num_of_features = net_params["num_of_features"]
        device = net_params["DEVICE"]
        node_feature_hidden1 = net_params["node_feature_hidden1"]
        self.graph_learn = GraphLearn(net_params["GLalpha"], self.num_of_features, device)
        self.cheb_conv = ChebConv(node_feature_hidden1, net_params["K"], self.num_of_features, device)

    def forward(self, x: torch.Tensor):
        s_matrix, adj_loss = self.graph_learn(x)
        gcn = self.cheb_conv([x, s_matrix])
        return gcn, s_matrix, adj_loss


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return x1, x2


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc_layer1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc_layer2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_layer1(x))
        return self.fc_layer2(x)


def aug_select_node(graph: torch.Tensor, drop_percent: float = 0.8) -> torch.Tensor:
    num = len(graph)
    select_num = int(num * drop_percent)
    all_node_list = [i for i in range(num)]
    selected_node_list = random.sample(all_node_list, select_num)
    index = torch.tensor(selected_node_list, dtype=torch.long, device=graph.device)
    return torch.index_select(graph, 0, index)


def aug_drop_node_list(graph_list: torch.Tensor, drop_percent: float) -> torch.Tensor:
    graph_num = len(graph_list)
    aug_list = []
    for i in range(graph_num):
        aug_list.append(aug_select_node(graph_list[i], drop_percent))
    aug = torch.stack(aug_list, 0)
    return torch.flatten(aug, start_dim=1, end_dim=-1)


class DSAGC(nn.Module):
    def __init__(self, net_params: dict):
        super().__init__()
        self.device = net_params["DEVICE"]
        out_feature = net_params["node_feature_hidden1"]
        channel = net_params["num_of_vertices"]
        linearsize = net_params["linearsize"]
        self.drop_rate = net_params["drop_rate"]
        self.gcn = GCNBlock(net_params)
        self.domain_classifier2 = DomainAdversarialLoss(hidden_1=64)
        self.domain_classifier3 = TripleDomainAdversarialLoss(hidden_1=64)
        self.domain_classifier2_3 = DomainAdversarialLossThreeAda(hidden_1=64)

        flat_dim = int(channel * net_params["num_of_features"])
        self.fea_extractor_f = FeatureExtractor(flat_dim, 64, 64)
        self.fea_extractor_g = FeatureExtractor(int(channel * self.drop_rate) * out_feature, linearsize, 64)
        self.fea_extractor_c = FeatureExtractor(64 * 2, 64, 32)

        self.projection_head = ProjectionHead(64, 16)
        self.classifier = nn.Linear(64, net_params["category_number"])
        self.self_attention = MultiHeadAttention(128, 128, 128, 128, 64, 0.5)

        self.batch_size = net_params["batch_size"]
        self.category_number = net_params["category_number"]
        self.multi_att = net_params["Multi_att"]

    def compute_diag_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        num = len(tensor)
        diag_sum = 0
        for i in range(num):
            diag_sum += tensor[i][i]
        return diag_sum

    def sim_matrix2(self, ori_vector: torch.Tensor, arg_vector: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        sim_tensor = None
        for i in range(len(ori_vector)):
            sim = torch.cosine_similarity(ori_vector[i].unsqueeze(0), arg_vector, dim=1) * (1 / temp)
            if i == 0:
                sim_tensor = sim.unsqueeze(0)
            else:
                sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
        return sim_tensor

    def forward(self, x: torch.Tensor, tripleada: int = 0, threshold: int = 0):
        feature, _, adj_loss = self.gcn(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extractor_f(feature1)

        if threshold:
            if tripleada:
                domain_output = self.domain_classifier3(feature1)
            else:
                domain_output = self.domain_classifier2_3(feature1)
        else:
            domain_output = self.domain_classifier2(feature1)

        aug_graph1 = aug_drop_node_list(feature, self.drop_rate)
        aug_graph2 = aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extractor_g(aug_graph1)
        _, aug_graph2_feature1 = self.fea_extractor_g(aug_graph2)

        aug_graph1_feature = self.projection_head(aug_graph1_feature1)
        aug_graph2_feature = self.projection_head(aug_graph2_feature1)

        l2_dist = torch.mean((aug_graph1_feature1 - aug_graph2_feature1) ** 2)

        sim_matrix_tmp2 = self.sim_matrix2(aug_graph1_feature, aug_graph2_feature, temp=1)
        row_softmax = nn.LogSoftmax(dim=1)
        row_softmax_matrix = -row_softmax(sim_matrix_tmp2)
        col_softmax = nn.LogSoftmax(dim=0)
        col_softmax_matrix = -col_softmax(sim_matrix_tmp2)

        row_diag_sum = self.compute_diag_sum(row_softmax_matrix)
        col_diag_sum = self.compute_diag_sum(col_softmax_matrix)
        contrastive_loss = (row_diag_sum + col_diag_sum) / (2 * len(row_softmax_matrix))

        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1).unsqueeze(1)
        if self.multi_att:
            class_feature = self.self_attention(class_feature, class_feature, class_feature)
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extractor_c(class_feature)
        pred = self.classifier(class_feature)

        s_feature = class_feature[: self.batch_size]
        t_feature = class_feature[-self.batch_size :]
        sim_sample = self.sim_matrix2(s_feature, t_feature)
        sim_weight = torch.mean(sim_sample, dim=1).unsqueeze(1)
        sim_weight = torch.softmax(sim_weight, dim=0)

        return pred, domain_output, adj_loss, contrastive_loss, sim_weight, l2_dist

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        feature, _, _ = self.gcn(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extractor_f(feature1)
        aug_graph1 = aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extractor_g(aug_graph1)
        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1).unsqueeze(1)
        if self.multi_att:
            class_feature = self.self_attention(class_feature, class_feature, class_feature)
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extractor_c(class_feature)
        pred = self.classifier(class_feature)
        return torch.softmax(pred, dim=1)


# Backward-compatible name with the original repo.
SemiGCL = DSAGC
