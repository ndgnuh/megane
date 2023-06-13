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
