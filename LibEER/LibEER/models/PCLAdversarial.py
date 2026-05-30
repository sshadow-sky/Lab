import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000.0, auto_step=False):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input_data):
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )
        if self.auto_step:
            self.step()
        return ReverseLayerF.apply(input_data, coeff)

    def step(self):
        self.iter_num += 1


class DAANLoss(nn.Module):
    def __init__(self, domain_classifier, num_class=3, reduction="mean", max_iter=1000):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.domain_classifier = domain_classifier
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True)
        self.local_classifiers = torch.nn.ModuleList()
        self.global_classifiers = domain_classifier
        for _ in range(num_class):
            self.local_classifiers.append(domain_classifier)

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_logits=None, target_logits=None):
        global_loss = self.get_global_adversarial_result(source, target)
        # Keep same behavior as original implementation: return global adversarial loss only.
        return global_loss

    def get_global_adversarial_result(self, f_s, f_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.global_classifiers(f)
        d_s, d_t = d.chunk(2, dim=0)

        d_label_s = torch.ones((f_s.size(0), 1), device=f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1), device=f_t.device)

        loss_s = F.binary_cross_entropy(d_s, d_label_s, reduction=self.reduction)
        loss_t = F.binary_cross_entropy(d_t, d_label_t, reduction=self.reduction)
        return 0.5 * (loss_s + loss_t)

    def get_local_adversarial_result(self, feat, logits, source=True):
        loss_adv = 0.0
        for c in range(self.num_class):
            x = feat[c + 1]
            x = self.grl(x)
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            logits_c = logits[:, c].reshape((softmax_logits.shape[0], 1))
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(x.size(0), 1).to(device)
            else:
                domain_label = torch.zeros(x.size(0), 1).to(device)
            loss_adv = loss_adv + F.binary_cross_entropy(domain_pred, domain_label, reduction=self.reduction)
        return 0.5 * loss_adv

    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0
