from collections import OrderedDict

import torch
from torch import nn


class MultiscaleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *a, **k):
        super().__init__()
        kernel_sizes = [3, 13, 31, 69]
        paddings = [k // 2 for k in kernel_sizes]
        self.num_convs = len(kernel_sizes) * 2
        assert out_channels % (self.num_convs) == 0
        aux_channels = out_channels // self.num_convs
        vconv = [
            nn.Conv2d(in_channels, aux_channels, (1, kern), padding=(0, p), **k)
            for p, kern in zip(paddings, kernel_sizes)
        ]
        hconv = [
            nn.Conv2d(in_channels, aux_channels, (kern, 1), padding=(p, 0), **k)
            for p, kern in zip(paddings, kernel_sizes)
        ]
        self.convs = nn.ModuleList(vconv + hconv)

    def forward(self, images):
        features = [conv(images) for conv in self.convs]
        return torch.cat(features, dim=1)


def Chain(*alayers, **klayers):
    modules = OrderedDict()
    for i, layer in enumerate(alayers):
        modules[str(i)] = layer
    for k, layer in klayers.items():
        modules[k] = layer
    return nn.Sequential(modules)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        act: callable = nn.ReLU,
        attention=None,
    ):
        super().__init__()
        if attention is None:
            self.attention = nn.MultiheadAttention(
                hidden_size,
                num_attention_heads,
                batch_first=True,
            )
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Forward attention
        residual = x
        x = self.norm_attention(x)
        x, _ = self.attention(x, x, x)
        x = x + residual

        # Forward mlp
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + residual

        return x


class AFAttention(nn.Module):
    def __init__(self, hidden_size: int, s: int):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.rand(1, s, hidden_size))

    def forward(self, x):
        # Project
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        eK = torch.exp(K + self.bias)
        sQ = torch.sigmoid(Q)
        ctx = (eK * V).sum(dim=1, keepdims=True) / eK.sum(dim=1, keepdims=True)
        ctx = sQ * ctx
        return ctx
