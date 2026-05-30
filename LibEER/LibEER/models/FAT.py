import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
from einops import rearrange


param_path = "config/model_param/FAT.yaml"


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation="gelu", use_p_bias=True):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2

        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_uniform_(self.input_linear_p.weight, nonlinearity="relu")
        init.kaiming_uniform_(self.input_linear_g.weight, nonlinearity="relu")
        if self.input_linear_p.bias is not None:
            init.zeros_(self.input_linear_p.bias)
        if self.input_linear_g.bias is not None:
            init.zeros_(self.input_linear_g.bias)

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)


def normalize_A(a, symmetry=False):
    a = torch.relu(a)
    if symmetry:
        a = a + torch.transpose(a, 0, 1)
    d = torch.sum(a, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    d_mat = torch.diag_embed(d)
    return torch.matmul(torch.matmul(d_mat, a), d_mat)


class ModifiedPatchEmbedding2D(nn.Module):
    def __init__(self, emb_size=40, num_channels=62, num_freq_bands=5):
        super(ModifiedPatchEmbedding2D, self).__init__()
        self.batch_norm_stage1 = nn.BatchNorm2d(emb_size // 2)
        self.batch_norm_stage2 = nn.BatchNorm2d(emb_size)

        self.position_encodings = nn.Parameter(torch.randn(1, 1, num_channels, num_freq_bands))

        self.conv2d_stage1 = nn.Sequential(
            nn.Conv2d(num_freq_bands, emb_size // 2, kernel_size=(1, 1)),
            self.batch_norm_stage1,
            nn.ReLU(),
        )

        self.conv2d_stage2 = nn.Sequential(
            nn.Conv2d(emb_size // 2, emb_size, kernel_size=(1, 1)),
            self.batch_norm_stage2,
            nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x + self.position_encodings
        x = x.squeeze(1).permute(0, 2, 1).unsqueeze(-1)

        x = self.conv2d_stage1(x)
        x = self.conv2d_stage2(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class DynamicGraphLearner(nn.Module):
    def __init__(self, n, num_heads=4):
        super(DynamicGraphLearner, self).__init__()
        self.num_heads = num_heads
        self.adj = nn.Parameter(torch.randn(n, n))
        nn.init.xavier_normal_(self.adj)

    def forward(self, x):
        b = x.size(0)
        adj_normalized = normalize_A(self.adj)
        adj_expanded = adj_normalized.unsqueeze(0).expand(b, -1, -1)
        adj_expanded = adj_expanded.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        return adj_expanded


class FAA(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5, use_dynamic_graph=False):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")

        self.keys = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.queries = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.values = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.use_dynamic_graph = use_dynamic_graph
        if self.use_dynamic_graph:
            self.dg_weight_linear1 = nn.Linear(emb_size // 2, 1)
            self.dg_weight_linear2 = nn.Linear(emb_size // 2, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        b, n, _ = x.shape
        h = self.num_heads

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=h)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=h)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=h)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        queries_4_4 = queries.permute(0, 2, 1, 3)
        q_front = queries_4_4[:, :, : h // 2, :]
        q_back = queries_4_4[:, :, h // 2 :, :]

        q_front_flat = q_front.reshape(b, n, -1)
        w_front = self.sigmoid(self.dg_weight_linear1(q_front_flat.reshape(-1, q_front_flat.shape[-1])))
        w_front = w_front.view(b, n, 1).unsqueeze(1).expand(-1, h // 2, -1, n)

        q_back_flat = q_back.reshape(b, n, -1)
        w_back = self.sigmoid(self.dg_weight_linear2(q_back_flat.reshape(-1, q_back_flat.shape[-1])))
        w_back = w_back.view(b, n, 1).unsqueeze(1).expand(-1, h // 2, -1, n)

        if dynamic_graph1 is not None:
            energy[:, : h // 2, :, :] = energy[:, : h // 2, :, :] + w_front * dynamic_graph1
        if dynamic_graph2 is not None:
            energy[:, h // 2 :, :, :] = energy[:, h // 2 :, :, :] + w_back * dynamic_graph2

        scaling = self.emb_size ** 0.5
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhqk, bhkd -> bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class AttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, drop_p, use_dynamic_graph=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(emb_size)
        self.faa = FAA(emb_size, num_heads, drop_p, use_dynamic_graph=use_dynamic_graph)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        x_norm = self.layernorm(x)
        out = self.faa(x_norm, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        out = self.dropout(out)
        return x + out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, use_dynamic_graph=False):
        super().__init__()
        self.attention = AttentionBlock(emb_size, num_heads, drop_p, use_dynamic_graph)
        self.feed_forward = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p),
                nn.Dropout(drop_p),
            )
        )

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        x = self.attention(x, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads=4, drop_p=0.2, num_dim=63, use_dynamic_graph=False):
        super().__init__()

        self.use_dynamic_graph = use_dynamic_graph
        self.dynamic_graph_learner1 = DynamicGraphLearner(n=num_dim, num_heads=num_heads // 2) if use_dynamic_graph else None
        self.dynamic_graph_learner2 = DynamicGraphLearner(n=num_dim, num_heads=num_heads // 2) if use_dynamic_graph else None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(emb_size, num_heads, drop_p, use_dynamic_graph=use_dynamic_graph)
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        dynamic_graph1 = self.dynamic_graph_learner1(x) if self.use_dynamic_graph else None
        dynamic_graph2 = self.dynamic_graph_learner2(x) if self.use_dynamic_graph else None

        for layer in self.layers:
            x = layer(x, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )


class FAT(nn.Module):
    def __init__(
        self,
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        emb_size=40,
        depth=6,
        num_heads=8,
        attention_dropout=0.2,
        position_dropout=0.1,
        forward_expansion=4,
        use_dynamic_graph=True,
    ):
        super(FAT, self).__init__()

        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.emb_size = emb_size
        self.depth = depth
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.position_dropout = position_dropout
        self.forward_expansion = forward_expansion
        self.use_dynamic_graph = use_dynamic_graph

        self.get_param()

        self.patch_embedding = ModifiedPatchEmbedding2D(self.emb_size, self.num_electrodes, self.in_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        self.positional_encoding = PositionalEncoding(
            self.emb_size,
            dropout=self.position_dropout,
            max_len=self.num_electrodes + 1,
        )
        self.transformer_encoder = TransformerEncoder(
            self.depth,
            self.emb_size,
            self.num_heads,
            drop_p=self.attention_dropout,
            num_dim=self.num_electrodes + 1,
            use_dynamic_graph=self.use_dynamic_graph,
        )
        self.classification_head = ClassificationHead(self.emb_size, self.num_classes)

    def get_param(self):
        try:
            with open(param_path, "r", encoding="utf-8") as fd:
                model_param = yaml.load(fd, Loader=yaml.FullLoader)
            params = model_param.get("params", {})
            self.emb_size = int(params.get("emb_size", self.emb_size))
            self.depth = int(params.get("depth", self.depth))
            self.num_heads = int(params.get("num_heads", self.num_heads))
            self.attention_dropout = float(params.get("attention_dropout", self.attention_dropout))
            self.position_dropout = float(params.get("position_dropout", self.position_dropout))
            self.forward_expansion = int(params.get("forward_expansion", self.forward_expansion))
            self.use_dynamic_graph = bool(params.get("use_dynamic_graph", self.use_dynamic_graph))
            print("\nUsing setting from {}\n".format(param_path))
        except IOError:
            print("\n{} may not exist or not available".format(param_path))

        print("FAT Model, Parameters:\n")
        print("{:45}{:20}".format("emb_size:", self.emb_size))
        print("{:45}{:20}".format("depth:", self.depth))
        print("{:45}{:20}".format("num_heads:", self.num_heads))
        print("{:45}{:20}".format("attention_dropout:", self.attention_dropout))
        print("{:45}{:20}".format("position_dropout:", self.position_dropout))
        print("{:45}{:20}".format("forward_expansion:", self.forward_expansion))
        print("{:45}{:20}\n".format("use_dynamic_graph:", self.use_dynamic_graph))

    def forward(self, x):
        x = self.patch_embedding(x)
        b, _, _ = x.shape
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]
        logits = self.classification_head(cls_output)
        return logits