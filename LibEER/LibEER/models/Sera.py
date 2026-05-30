# models/Sera.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()

        if activation:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Patcher(nn.Module):
    def __init__(self, patch_size, stride, in_chan, out_dim):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        patch_dim = int(in_chan * patch_size[0] * patch_size[1])

        if out_dim == patch_dim:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Linear(patch_dim, out_dim, bias=False)

    def forward(self, x):
        # x: [B, K, C, T]
        x = self.unfold(x)          # [B, patch_dim, num_patches]
        x = x.transpose(1, 2)       # [B, num_patches, patch_dim]
        x = self.to_out(x)
        return x


class MMDVAE(nn.Module):
    """
    Multiple Multi-stage Decoder VAE.
    """

    def __init__(self, dimx, dimz, n_sources=3, variational=True):
        super().__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.variational = variational

        chans = (128, 64, self.dimz)

        self.out_z = nn.Linear(chans[-1], 2 * self.n_sources * self.dimz)

        self.encoder = nn.Sequential(
            LinearBlock(self.dimx, chans[0]),
            LinearBlock(chans[0], chans[1]),
            LinearBlock(chans[1], chans[2]),
        )

        self.decoder_lv1 = nn.Sequential(
            LinearBlock(self.dimz, chans[-1]),
            LinearBlock(chans[-1], self.dimz),
        )

        self.decoder_lv2 = nn.Sequential(
            LinearBlock(chans[2], chans[1]),
            LinearBlock(chans[1], chans[0]),
            LinearBlock(chans[0], self.dimx, activation=False),
        )

    def encode(self, x):
        d = self.encoder(x)
        dz = self.out_z(d)

        mu = dz[:, ::2]
        logvar = dz[:, 1::2]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

        return z

    def decode(self, z):
        recon_separate = torch.sigmoid(z).view(-1, self.n_sources, self.dimz)

        b, s, d = recon_separate.shape

        recon_separate_flat = recon_separate.reshape(b * s, d)
        recon_separate_flat = self.decoder_lv1(recon_separate_flat)
        recon_separate = recon_separate_flat.reshape(b, s, d)

        recon_x = recon_separate.sum(dim=1)
        recon_x = self.decoder_lv2(recon_x)

        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x)

        if self.variational:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        recon_x, recons = self.decode(z)

        return recon_x, mu, logvar, recons


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        # x: [B, N, D]
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = [
            t.reshape(t.shape[0], t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
            for t in qkv
        ]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.heads * self.dim_head)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )

        return super().forward(x)


class EncoderTSc(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        super().__init__()

        self.inception_window = [0.5, 0.375, 0.25, 0.125]
        self.pool = 2

        self.Tception1 = self.conv_block(
            1,
            int(num_T),
            (1, int(self.inception_window[0] * sampling_rate + 1)),
            1,
            self.pool,
            self.get_padding(int(self.inception_window[0] * sampling_rate + 1)),
        )

        self.Tception2 = self.conv_block(
            1,
            int(num_T),
            (1, int(self.inception_window[1] * sampling_rate + 1)),
            1,
            self.pool,
            self.get_padding(int(self.inception_window[1] * sampling_rate + 1)),
        )

        self.Tception3 = self.conv_block(
            1,
            int(num_T),
            (1, int(self.inception_window[2] * sampling_rate + 1)),
            1,
            self.pool,
            self.get_padding(int(self.inception_window[2] * sampling_rate + 1)),
        )

        self.Tception4 = self.conv_block(
            1,
            int(num_T),
            (1, int(self.inception_window[3] * sampling_rate + 1)),
            1,
            self.pool,
            self.get_padding(int(self.inception_window[3] * sampling_rate + 1)),
        )

        self.Tfusion = self.conv_block(num_T * 4, num_T, (1, 1), 1, int(self.pool * 0.5))
        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.5))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y

        y = self.Tception2(x)
        out = torch.cat((out, y), dim=1)

        y = self.Tception3(x)
        out = torch.cat((out, y), dim=1)

        y = self.Tception4(x)
        out = torch.cat((out, y), dim=1)

        out = self.Tfusion(out)
        out = self.BN_t(out)

        z = self.Sception1(out)
        out = self.BN_s(z)
        out = self.fc(out)

        return out

    def get_padding(self, kernel):
        return 0, int(0.5 * (kernel - 1))


class Sera(nn.Module):
    """
    LibEER-compatible Sera model.

    Input:
        x: [B, 1, channels, time_points]

    Output during normal classification:
        logits: [B, num_classes]

    Output with return_all=True:
        y, y_rec, z_source, logits, domain_logits, dta, out_tsne
    """

    def __init__(
        self,
        num_classes,
        input_size,
        sampling_rate=128,
        num_T=32,
        patch_size=16,
        patch_stride=8,
        dropout_rate=0.25,
        pool=2,
        dimz=32,
        m=3,
        transformer_depth=2,
        num_head=16,
        return_all=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.sampling_rate = sampling_rate
        self.num_T = num_T
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dropout_rate = dropout_rate
        self.pool = pool
        self.dimz = dimz
        self.m = m
        self.transformer_depth = transformer_depth
        self.num_head = num_head
        self.return_all = return_all

        if len(input_size) != 3:
            raise ValueError(f"input_size should be (1, channels, time_points), got {input_size}")

        self.channel = input_size[1]

        if 0.5 * sampling_rate % 2 == 0:
            kernel_size = (1, int(0.5 * sampling_rate + 1))
        else:
            kernel_size = (1, int(0.5 * sampling_rate))

        self.encodertsc = EncoderTSc(
            num_classes=num_classes,
            input_size=input_size,
            sampling_rate=sampling_rate,
            num_T=num_T,
            num_S=num_T,
            hidden=dimz * 2,
            dropout_rate=dropout_rate,
        )

        self.encoders = nn.Sequential(
            Conv2dWithConstraint(
                input_size[0],
                num_T,
                kernel_size,
                padding=self.get_padding(kernel_size[-1]),
                max_norm=2,
            ),
            Conv2dWithConstraint(
                num_T,
                num_T,
                (input_size[-2], 1),
                padding=0,
                max_norm=2,
            ),
            nn.BatchNorm2d(num_T),
            nn.ELU(),
            nn.MaxPool2d((1, self.pool), stride=(1, self.pool)),
        )

        self.patch_size_2d = [1, patch_size]
        self.patch_stride_2d = [1, patch_stride]

        reduced_t = int(input_size[-1] / self.pool)

        if (reduced_t - patch_size) < 0:
            raise ValueError(
                f"patch_size={patch_size} is larger than reduced time length={reduced_t}. "
                f"Please reduce --sera_patch_size."
            )

        self.in_chan = num_T
        self.out_dim = int(num_T * self.patch_size_2d[0] * self.patch_size_2d[1])

        self.patcher = Patcher(
            self.patch_size_2d,
            self.patch_stride_2d,
            self.in_chan,
            self.out_dim,
        )

        self.seq = int((reduced_t - patch_size) // patch_stride + 1)

        self.dimx = int(num_T * self.patch_size_2d[0] * self.patch_size_2d[1])

        self.vae = MMDVAE(self.dimx, self.dimz, self.m)

        self.domain_classifier = nn.Sequential(
            nn.Linear(int(self.seq * self.m * self.dimz), int(self.seq * self.m)),
            nn.ReLU(),
            nn.Linear(int(self.seq * self.m), 2),
        )

        self.encodert = nn.Sequential(
            nn.Linear(int(self.m * self.dimz), int(self.dimz)),
            nn.ReLU(),
            nn.Linear(int(self.dimz), int(self.dimz)),
        )

        self.transformer = Transformer(
            self.dimz,
            self.transformer_depth,
            self.num_head,
            self.dimz,
            self.dimz,
            dropout_rate,
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.seq * self.dimz), num_classes),
        )

        # Kept for consistency with original implementation, but not used in final loss.
        self.decoder1 = nn.Sequential(
            nn.Linear(int(input_size[-1]), int(self.m * self.dimz)),
            nn.ReLU(),
        )

    def forward_features(self, x, alpha=1.0):
        y = self.encodertsc(x)
        y = self.patcher(y)

        y_flat = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        y_rec, mu, logvar, z_source = self.vae(y_flat)

        # z_source: [(B*seq), m, dimz]
        z_s_d = z_source.reshape(z_source.shape[0], self.m * self.dimz)

        # z_t: [B, seq, m*dimz]
        z_t = z_s_d.reshape(x.size(0), self.seq, self.m * self.dimz)

        # domain feature: [B, seq*m*dimz]
        z_domain = z_t.reshape(x.size(0), self.seq * self.m * self.dimz)

        reverse_feature = ReverseLayerF.apply(z_domain, alpha)
        domain_logits = self.domain_classifier(reverse_feature)

        dta = self.encodert(z_t)

        out = self.transformer(dta) + dta
        out_tsne = out

        out_flat = out.reshape(x.size(0), self.seq * self.dimz)
        logits = self.fc(out_flat)

        return y_flat, y_rec, z_source, logits, domain_logits, dta, out_tsne

    def forward(self, x, alpha=1.0, return_all=None):
        if return_all is None:
            return_all = self.return_all

        y, y_rec, z_source, logits, domain_logits, dta, out_tsne = self.forward_features(x, alpha)

        if return_all:
            return y, y_rec, z_source, logits, domain_logits, dta, out_tsne

        return logits

    def get_padding(self, kernel):
        return 0, int(0.5 * (kernel - 1))


# Alias for compatibility with the original repository naming.
SERA = Sera


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def covariance_matrix(x):
    """
    x: [B, seq, hidden]
    """
    if x.size(1) < 2:
        return torch.zeros(
            x.size(0),
            x.size(2),
            x.size(2),
            device=x.device,
            dtype=x.dtype,
        )

    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    x_centered = x - x.mean(dim=1, keepdim=True)

    cov_matrix = torch.matmul(
        x_centered.transpose(1, 2),
        x_centered,
    ) / max(x_centered.size(1) - 1, 1)

    cov_matrix = torch.nan_to_num(cov_matrix, nan=0.0, posinf=1e4, neginf=-1e4)
    return cov_matrix


def temporal_alignment_loss(p, q):
    """
    p, q: [B, seq, hidden]
    """
    if p.size(1) < 2 or q.size(1) < 2:
        return p.new_tensor(0.0)

    cov_p = covariance_matrix(p)
    cov_q = covariance_matrix(q)

    diff = cov_p - cov_q
    loss = torch.norm(diff, p="fro", dim=(1, 2))
    loss = torch.mean(loss)

    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
    return loss