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


class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
        self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = x.reshape(-1, self.input_dim)
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear) + self.u_linear, dim=1)
        res = torch.mul(attn_softmax, x_reshape)
        res = res.reshape(batch_size, time_steps, self.input_dim)
        return res


class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=1, location=-1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location

    def forward(self, x):
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn


class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=1):
        super().__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)

    def forward(self, x):
        return self.theta(x)


class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=1, output_dim=310):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_data, hidden, cell, time_steps):
        out = []
        out_cur = self.fc_out(input_data)
        out.append(out_cur)
        out_cur = out_cur.unsqueeze(0)
        for _ in range(time_steps - 1):
            output, (hidden, cell) = self.rnn(out_cur, (hidden, cell))
            out_cur = self.fc_out(output.squeeze(0))
            out.append(out_cur)
            out_cur = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1, 0)
        return out, hidden, cell


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=14):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.classifier(x)


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


def time_steps_shuffle(source_data):
    source_data_1 = source_data.clone()
    cur_time_step = source_data_1[:, -1, :]
    dim_size = source_data[:, :-1, :].size(1)
    idxs = list(range(dim_size))
    random.shuffle(idxs)
    else_part = source_data_1[:, idxs, :]
    result = torch.cat([else_part, cur_time_step.unsqueeze(1)], dim=1)
    return result


class DMMRPreTrainingModel(nn.Module):
    def __init__(self, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15, input_dim=310, hid_dim=64, n_layers=1):
        super().__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(input_dim=input_dim)
        self.sharedEncoder = Encoder(input_dim=input_dim, hid_dim=hid_dim, n_layers=n_layers)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=hid_dim, output_dim=number_of_source)
        self.decoders = nn.ModuleList(
            [Decoder(input_dim=input_dim, hid_dim=hid_dim, n_layers=n_layers, output_dim=input_dim) for _ in range(number_of_source)]
        )

    def forward(self, x, corres, subject_id, m=0.0, mark=0):
        del mark
        x = time_steps_shuffle(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict, dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)

        rec_loss = 0
        mix_subject_feature = 0
        for decoder in self.decoders:
            x_out, *_ = decoder(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mix_subject_feature += x_out

        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mix_subject_feature)
        for i, decoder in enumerate(self.decoders):
            x_out, *_ = decoder(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])

        return rec_loss, sim_loss


class DMMRFineTuningModel(nn.Module):
    def __init__(self, base_model, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super().__init__()
        self.baseModel = copy.deepcopy(base_model)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = self.baseModel.attentionLayer
        self.sharedEncoder = self.baseModel.sharedEncoder
        self.cls_fc = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, number_of_category, bias=True),
        )
        self.mse = MSE()

    def forward(self, x, label_src=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, _, _ = self.sharedEncoder(x)
        x_logits = self.cls_fc(shared_last_out)
        x_pred = F.log_softmax(x_logits, dim=1)
        cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        return x_pred, x_logits, cls_loss


class DMMRTestModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.baseModel = copy.deepcopy(base_model)

    def forward(self, x):
        x = self.baseModel.attentionLayer(x, x.shape[0], self.baseModel.time_steps)
        shared_last_out, _, _ = self.baseModel.sharedEncoder(x)
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