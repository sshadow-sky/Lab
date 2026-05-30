import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from sklearn import metrics
from torch.autograd import Function


class PAACLoss(object):
    def __init__(
        self,
        num_layers,
        kernel_num,
        kernel_mul,
        num_classes,
        threshold,
        low_rank,
        hidden_2,
        hidden_4,
        intra_only=False,
    ):
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes == 1)
        self.num_layers = num_layers
        self.filtered_classes = []
        self.threshold = threshold
        self.class_num_min = 2
        self.cluster_label = np.arange(num_classes, dtype=np.int64)
        self.P_tar = torch.zeros(num_classes, hidden_4)
        self.proto_dist = torch.tensor(0.0)

    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []

        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list.append(dist_c)

        return dist_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist["ss"]) + torch.sum(dist["tt"]) + 2 * torch.sum(dist["st"])

        bs_s = dist["ss"].size(0)
        bs_t = dist["tt"].size(0)

        n = bs_s * bs_s + bs_t * bs_t + 2 * bs_s * bs_t - bs_s - bs_t
        if n <= 0:
            return 1.0

        gamma = dist_sum.item() / n
        if gamma <= 0:
            gamma = 1.0

        return gamma

    def patch_gamma_estimation(self, nums_s, nums_t, dist):
        assert len(nums_s) == len(nums_t)

        num_classes = len(nums_s)
        device = dist["st"].device
        dtype = dist["st"].dtype

        patch = {}
        gammas = {}

        gammas["st"] = torch.zeros_like(dist["st"], requires_grad=False)
        gammas["ss"] = []
        gammas["tt"] = []

        for _ in range(num_classes):
            gammas["ss"].append(torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=False))
            gammas["tt"].append(torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=False))

        source_start = source_end = 0

        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_s[ns]
            patch["ss"] = dist["ss"][ns]

            target_start = target_end = 0

            for nt in range(num_classes):
                target_start = target_end
                target_end = target_start + nums_t[nt]
                patch["tt"] = dist["tt"][nt]

                patch["st"] = dist["st"].narrow(0, source_start, nums_s[ns]).narrow(
                    1, target_start, nums_t[nt]
                )

                gamma = self.gamma_estimation(patch)

                gammas["ss"][ns][nt] = gamma
                gammas["tt"][nt][ns] = gamma
                gammas["st"][source_start:source_end, target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        device = dist.device
        dtype = dist.dtype

        gamma = torch.as_tensor(gamma, device=device, dtype=dtype)

        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul ** i) for i in range(kernel_num)]
        gamma_tensor = torch.stack(gamma_list, dim=0).to(device=device, dtype=dtype)

        eps = 1e-5
        gamma_tensor = torch.clamp(gamma_tensor, min=eps).detach()

        while gamma_tensor.dim() > dist.dim():
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor

        dist = torch.clamp(dist, min=1e-5, max=1e5)

        kernel_val = torch.sum(torch.exp(-dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        kernel_dist = None

        for i in range(self.num_layers):
            if category is None:
                dist = dist_layers[i][key]
                gamma = gamma_layers[i][key]
            else:
                dist = dist_layers[i][key][category]
                gamma = gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            cur_kernel_dist = self.compute_kernel_dist(
                dist,
                gamma,
                cur_kernel_num,
                cur_kernel_mul,
            )

            if kernel_dist is None:
                kernel_dist = cur_kernel_dist
            else:
                kernel_dist = kernel_dist + cur_kernel_dist

        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert len(nums_row) == len(nums_col)

        num_classes = len(nums_row)
        device = dist.device
        dtype = dist.dtype

        mean_tensor = torch.zeros(num_classes, num_classes, device=device, dtype=dtype)

        row_start = row_end = 0

        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0

            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]

                block = dist.narrow(0, row_start, nums_row[row]).narrow(
                    1, col_start, nums_col[col]
                )

                mean_tensor[row, col] = torch.mean(block)

        return mean_tensor

    def compute_paired_dist(self, a, b):
        bs_a = a.size(0)
        bs_b = b.size(0)
        feat_len = a.size(1)

        a_expand = a.unsqueeze(1).expand(bs_a, bs_b, feat_len)
        b_expand = b.unsqueeze(0).expand(bs_a, bs_b, feat_len)

        dist = ((a_expand - b_expand) ** 2).sum(2)
        return dist

    def cal_PAAC_loss(self, source, target, nums_s, nums_t):
        assert len(nums_s) == len(nums_t), (
            "The number of classes for source (%d) and target (%d) should be the same."
            % (len(nums_s), len(nums_t))
        )

        if len(nums_s) == 0:
            return source.new_tensor(0.0)

        num_classes = len(nums_s)

        dist_layers = []
        gamma_layers = []

        for _ in range(self.num_layers):
            cur_source = source
            cur_target = target

            dist = {}
            dist["ss"] = self.compute_paired_dist(cur_source, cur_source)
            dist["tt"] = self.compute_paired_dist(cur_target, cur_target)
            dist["st"] = self.compute_paired_dist(cur_source, cur_target)

            dist["ss"] = self.split_classwise(dist["ss"], nums_s)
            dist["tt"] = self.split_classwise(dist["tt"], nums_t)

            dist_layers.append(dist)
            gamma_layers.append(self.patch_gamma_estimation(nums_s, nums_t, dist))

        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]["ss"][c] = gamma_layers[i]["ss"][c].view(num_classes, 1, 1)
                gamma_layers[i]["tt"][c] = gamma_layers[i]["tt"][c].view(num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, "st")
        kernel_dist_st = self.patch_mean(nums_s, nums_t, kernel_dist_st)

        kernel_dist_ss = []
        kernel_dist_tt = []

        for c in range(num_classes):
            ss_c = self.kernel_layer_aggregation(dist_layers, gamma_layers, "ss", c)
            tt_c = self.kernel_layer_aggregation(dist_layers, gamma_layers, "tt", c)

            kernel_dist_ss.append(torch.mean(ss_c.view(num_classes, -1), dim=1))
            kernel_dist_tt.append(torch.mean(tt_c.view(num_classes, -1), dim=1))

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st

        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / num_classes

        inter = None

        if not self.intra_only and num_classes > 1:
            device = mmds.device
            mask = (torch.ones(num_classes, num_classes, device=device) - torch.eye(num_classes, device=device)).bool()
            inter_mmds = torch.masked_select(mmds, mask)
            inter = torch.sum(inter_mmds) / (num_classes * (num_classes - 1))

        paa_c = intra if inter is None else intra - inter
        return paa_c

    def update_cluster_label(self, cluster_label):
        self.cluster_label = np.asarray(cluster_label, dtype=np.int64)

    def update_PAAC_threshold(self, threshold):
        self.threshold = threshold

    def fea_label_sort(self, source, target, s_label, t_label):
        sorted_src_fea = []
        sorted_src_labels_num = []
        sorted_tar_fea = []
        sorted_tar_labels_num = []

        for cls in self.filtered_classes:
            cls_index = torch.where(s_label == cls)[0]
            cls_index_tar = torch.where(t_label == cls)[0]

            # source 或 target 某类为空时，跳过该类，避免后面 gamma / MMD 出错
            if cls_index.numel() == 0 or cls_index_tar.numel() == 0:
                continue

            sorted_src_fea.append(source[cls_index])
            sorted_src_labels_num.append(cls_index.size(0))

            sorted_tar_fea.append(target[cls_index_tar])
            sorted_tar_labels_num.append(cls_index_tar.size(0))

        if len(sorted_src_fea) == 0:
            return (
                source.new_empty((0, source.size(1))),
                target.new_empty((0, target.size(1))),
                [],
                [],
            )

        sorted_src_fea = torch.cat(sorted_src_fea, dim=0)
        sorted_tar_fea = torch.cat(sorted_tar_fea, dim=0)

        return sorted_src_fea, sorted_tar_fea, sorted_src_labels_num, sorted_tar_labels_num

    def filter_classes(self, t_label):
        self.filtered_classes = []

        if t_label.numel() == 0:
            return

        for c in range(self.num_classes):
            mask = t_label == c
            count = torch.sum(mask).item()

            if count >= self.class_num_min:
                self.filtered_classes.append(c)

    def filter_samples(self, target, t_label):
        labels_index = torch.argmax(t_label, dim=1)

        labels_single = t_label.gather(1, labels_index.view(-1, 1)).squeeze(1)
        mask = labels_single >= self.threshold

        cluster_label_tensor = torch.as_tensor(
            self.cluster_label,
            device=target.device,
            dtype=labels_index.dtype,
        )

        labels_index = cluster_label_tensor[labels_index]

        if mask.sum().item() == 0:
            filtered_feature = target.new_empty((0, target.size(1)))
            filtered_label_single = labels_index.new_empty((0,))
            filtered_label = t_label.new_empty((0, t_label.size(1)))
        else:
            filtered_feature = target[mask]
            filtered_label_single = labels_index[mask]
            filtered_label = t_label[mask]

        return filtered_feature, filtered_label_single, filtered_label_single.size(0), filtered_label

    def get_loss(self, source, target, s_label, t_label, source_proto):
        s_label_single = torch.argmax(s_label, dim=1)

        target, t_label_single, selected_num, t_label = self.filter_samples(target, t_label)

        self.filter_classes(t_label_single)
        self.compute_target_proto(target, t_label_single, source_proto)

        source, target, nums_s, nums_t = self.fea_label_sort(
            source,
            target,
            s_label_single,
            t_label_single,
        )

        if len(nums_t) == 0:
            return source.new_tensor(0.0), selected_num, source.new_tensor(0.0)

        paa_c = self.cal_PAAC_loss(source, target, nums_s, nums_t)

        return paa_c, selected_num, self.proto_dist.detach().cpu()

    def compute_target_proto(self, target, t_label_single, source_proto):
        device = target.device

        if target.size(0) == 0 or t_label_single.size(0) == 0:
            self.P_tar = torch.zeros(
                self.num_classes,
                source_proto.size(1),
                device=device,
                dtype=source_proto.dtype,
            )
            self.proto_dist = torch.zeros(self.num_classes, device=device, dtype=source_proto.dtype)
            return

        t_label_single = t_label_single.long().to(device)
        source_proto = source_proto.to(device)

        if len(self.filtered_classes) == self.num_classes:
            eye = torch.eye(self.num_classes, device=device, dtype=target.dtype)

            t_label = eye[t_label_single]

            self.P_tar = torch.matmul(
                torch.inverse(torch.diag(t_label.sum(axis=0)) + eye),
                torch.matmul(t_label.T, target),
            )

            self.proto_dist = self.compute_paired_dist(self.P_tar, source_proto).diag()
        else:
            self.P_tar = torch.zeros(
                self.num_classes,
                source_proto.size(1),
                device=device,
                dtype=source_proto.dtype,
            )
            self.proto_dist = torch.zeros(self.num_classes, device=device, dtype=source_proto.dtype)


class PAALLoss(nn.Module):
    def __init__(self, class_num, kernel_type="rbf", kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(PAALLoss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)),
            int(total.size(0)),
            int(total.size(1)),
        )

        total1 = total.unsqueeze(1).expand(
            int(total.size(0)),
            int(total.size(0)),
            int(total.size(1)),
        )

        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            denom = n_samples ** 2 - n_samples
            bandwidth = torch.sum(l2_distance.detach()) / max(denom, 1)

        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth = torch.clamp(bandwidth, min=1e-5)

        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size(0)
        device = source.device

        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label,
            t_label,
            batch_size=batch_size,
            class_num=self.class_num,
        )

        weight_ss = torch.from_numpy(weight_ss).to(device)
        weight_tt = torch.from_numpy(weight_tt).to(device)
        weight_st = torch.from_numpy(weight_st).to(device)

        loss = source.new_tensor(0.0)

        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )

        if torch.isnan(kernels).any():
            return loss

        ss = kernels[:batch_size, :batch_size]
        tt = kernels[batch_size:, batch_size:]
        st = kernels[:batch_size, batch_size:]

        loss = loss + torch.sum(weight_ss * ss + weight_tt * tt - 2 * weight_st * st)

        return loss

    def convert_to_onehot(self, sca_label, class_num):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size, class_num):
        batch_size = s_label.size(0)
        batch_size_target = t_label.size(0)

        s_sca_label = s_label.detach().cpu().data.max(1)[1].numpy()
        s_vec_label = s_label.detach().cpu().data.numpy()
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.detach().cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.detach().cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))

        mask_arr_s = np.zeros((batch_size, class_num))
        mask_arr_t = np.zeros((batch_size_target, class_num))

        mask_arr_s[:, index] = 1
        mask_arr_t[:, index] = 1

        t_vec_label = t_vec_label * mask_arr_t
        s_vec_label = s_vec_label * mask_arr_s

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)

        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        return (
            weight_ss.astype("float32"),
            weight_tt.astype("float32"),
            weight_st.astype("float32"),
        )


class feature_extractor(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2):
        super(feature_extractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]


class discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_parameters(self) -> List[Dict]:
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]


class Pairwise_Learning(nn.Module):
    def __init__(
        self,
        hidden_2,
        hidden_4,
        num_of_class,
        low_rank,
        max_iter,
        upper_threshold,
        lower_threshold,
        P,
    ):
        super(Pairwise_Learning, self).__init__()

        self.max_iter = max_iter
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.threshold = upper_threshold

        self.cluster_label = np.arange(num_of_class, dtype=np.int64)
        self.num_of_class = num_of_class

        self.V = nn.Parameter(torch.randn(low_rank, hidden_4), requires_grad=True)
        self.U = nn.Parameter(torch.randn(low_rank, hidden_2), requires_grad=True)

        self.register_buffer("stored_mat", torch.zeros(low_rank, num_of_class))

    def forward(self, target_feature, P):
        P = P.to(target_feature.device)
        self.stored_mat = torch.matmul(self.V, P.T)

        target_predict = torch.matmul(
            torch.matmul(self.U, target_feature.T).T,
            self.stored_mat,
        )

        target_label_feature = torch.softmax(target_predict, dim=1)
        sim_matrix_target = self.get_cos_similarity_distance(target_label_feature)

        return sim_matrix_target, target_label_feature

    def target_domain_evaluation(self, feature_target_f, test_labels):
        self.eval()

        device = feature_target_f.device

        with torch.no_grad():
            test_logit = torch.matmul(
                torch.matmul(self.U, feature_target_f.T).T,
                self.stored_mat.to(device),
            )

            test_cluster = torch.argmax(torch.softmax(test_logit, dim=1), dim=1).detach().cpu().numpy()

        if test_labels.dim() == 2:
            test_labels = torch.argmax(test_labels, dim=1)

        test_labels = test_labels.detach().cpu().numpy()

        for i in range(len(self.cluster_label)):
            samples_in_cluster_index = np.where(test_cluster == i)[0]
            label_for_samples = test_labels[samples_in_cluster_index]

            if len(label_for_samples) == 0:
                self.cluster_label[i] = i
            else:
                self.cluster_label[i] = np.argmax(np.bincount(label_for_samples.astype(np.int64)))

        test_predict = np.zeros_like(test_labels)

        for i in range(len(self.cluster_label)):
            cluster_index = np.where(test_cluster == i)[0]
            test_predict[cluster_index] = self.cluster_label[i]

        acc = np.sum(test_predict == test_labels) / len(test_predict)
        nmi = metrics.normalized_mutual_info_score(test_predict, test_labels)

        return acc, nmi

    def predict(self, feature_target_f):
        with torch.no_grad():
            self.eval()

            device = feature_target_f.device

            test_logit = torch.matmul(
                torch.matmul(self.U, feature_target_f.T).T,
                self.stored_mat.to(device),
            ) / 8.0

            test_cluster = torch.argmax(torch.softmax(test_logit, dim=1), dim=1).detach().cpu().numpy()

            test_predict = np.zeros_like(test_cluster)

            for c in range(self.num_of_class):
                idx = np.where(test_cluster == c)[0]
                test_predict[idx] = self.cluster_label[c]

        return test_predict

    def get_cos_similarity_distance(self, features):
        features_norm = torch.norm(features, dim=1, keepdim=True).clamp_min(1e-12)
        features = features / features_norm
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype

        similar = torch.tensor(1.0, dtype=dtype, device=device)
        dissimilar = torch.tensor(0.0, dtype=dtype, device=device)

        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar, dissimilar)
        return sim_matrix

    def compute_indicator(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype

        selected = torch.tensor(1.0, dtype=dtype, device=device)
        not_selected = torch.tensor(0.0, dtype=dtype, device=device)

        w2 = torch.where(cos_dist_matrix < self.lower_threshold, selected, not_selected)
        w1 = torch.where(cos_dist_matrix > self.upper_threshold, selected, not_selected)

        w = w1 + w2
        nb_selected = torch.sum(w)

        return w, nb_selected

    def update_threshold(self, epoch: int):
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / max(n_epochs, 1)

        if epoch != 0:
            self.upper_threshold = self.upper_threshold - eta
            self.lower_threshold = self.lower_threshold + eta

        self.threshold = (self.upper_threshold + self.lower_threshold) / 2

    def get_parameters(self) -> List[Dict]:
        return [
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
        ]


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(
        self,
        alpha: Optional[float] = 1.0,
        lo: Optional[float] = 0.0,
        hi: Optional[float] = 1.0,
        max_iters: Optional[int] = 1000,
        auto_step: Optional[bool] = False,
    ):
        super(WarmStartGradientReverseLayer, self).__init__()

        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )

        if self.auto_step:
            self.step()

        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = "mean", max_iter=1000):
        super(DomainAdversarialLoss, self).__init__()

        self.grl = WarmStartGradientReverseLayer(
            alpha=1.0,
            lo=0.0,
            hi=1.0,
            max_iters=max_iter,
            auto_step=True,
        )

        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)

        d_s, d_t = d.chunk(2, dim=0)

        d_label_s = torch.ones((f_s.size(0), 1), device=f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1), device=f_t.device)

        self.domain_discriminator_accuracy = 0.5 * (
            binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t)
        )

        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class PAA(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_1,
        hidden_2,
        hidden_4,
        num_of_class,
        low_rank,
        max_iter,
    ):
        super(PAA, self).__init__()

        self.fea_extrator_f = feature_extractor(input_dim, hidden_1, hidden_2)

        self.U = nn.Parameter(torch.randn(low_rank, hidden_2), requires_grad=True)
        self.V = nn.Parameter(torch.randn(low_rank, hidden_4), requires_grad=True)

        self.register_buffer("P", torch.randn(num_of_class, hidden_4))
        self.register_buffer("stored_mat", torch.zeros(low_rank, num_of_class))

        self.max_iter = max_iter
        self.cluster_label = np.arange(num_of_class, dtype=np.int64)
        self.num_of_class = num_of_class
        self.hidden_2 = hidden_2
        self.hidden_4 = hidden_4

        if hidden_2 != hidden_4:
            raise ValueError(
                f"Current PAA implementation requires hidden_2 == hidden_4, "
                f"but got hidden_2={hidden_2}, hidden_4={hidden_4}. "
                f"Please set -hidden_2 and -hidden_4 to the same value, e.g. 64."
            )

    def forward(self, source, target, source_label):
        feature_source_f = self.fea_extrator_f(source)
        feature_target_f = self.fea_extrator_f(target)

        feature_source_g = feature_source_f

        device = source_label.device
        dtype = feature_source_g.dtype

        eye = torch.eye(self.num_of_class, device=device, dtype=dtype)

        self.P = torch.matmul(
            torch.inverse(torch.diag(source_label.sum(axis=0)) + eye),
            torch.matmul(source_label.T, feature_source_g),
        )

        self.stored_mat = torch.matmul(self.V, self.P.T)

        source_predict = torch.matmul(
            torch.matmul(self.U, feature_source_f.T).T,
            self.stored_mat,
        )

        source_label_feature = torch.softmax(source_predict, dim=1)
        sim_matrix_source = self.get_cos_similarity_distance(source_label_feature)

        return source_predict, feature_source_f, feature_target_f, sim_matrix_source

    def compute_target_centroid(self, target, target_label):
        feature_target = self.fea_extrator_f(target)

        device = target_label.device
        dtype = feature_target.dtype

        eye = torch.eye(self.num_of_class, device=device, dtype=dtype)

        target_centroid = torch.matmul(
            torch.inverse(torch.diag(target_label.sum(axis=0)) + eye),
            torch.matmul(target_label.T, feature_target),
        )

        return target_centroid

    def cluster_label_update(self, source_features, source_labels):
        self.eval()

        with torch.no_grad():
            feature_source_f = self.fea_extrator_f(source_features)

            source_logit = torch.matmul(
                torch.matmul(self.U, feature_source_f.T).T,
                self.stored_mat.to(feature_source_f.device),
            )

            source_cluster = torch.argmax(torch.softmax(source_logit, dim=1), dim=1).detach().cpu().numpy()

        if source_labels.dim() == 2:
            source_labels = torch.argmax(source_labels, dim=1)

        source_labels = source_labels.detach().cpu().numpy()

        for i in range(len(self.cluster_label)):
            samples_in_cluster_index = np.where(source_cluster == i)[0]
            label_for_samples = source_labels[samples_in_cluster_index]

            if len(label_for_samples) == 0:
                self.cluster_label[i] = i
            else:
                self.cluster_label[i] = np.argmax(np.bincount(label_for_samples.astype(np.int64)))

        source_predict = np.zeros_like(source_labels)

        for i in range(len(self.cluster_label)):
            cluster_index = np.where(source_cluster == i)[0]
            source_predict[cluster_index] = self.cluster_label[i]

        acc = np.sum(source_predict == source_labels) / len(source_predict)
        nmi = metrics.normalized_mutual_info_score(source_predict, source_labels)

        return acc, nmi

    def get_cos_similarity_distance(self, features):
        features_norm = torch.norm(features, dim=1, keepdim=True).clamp_min(1e-12)
        features = features / features_norm
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_parameters(self) -> List[Dict]:
        return [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
        ]