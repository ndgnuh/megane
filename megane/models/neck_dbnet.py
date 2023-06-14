from typing import List

import torch
from torch import nn


def ConvUpsample(hidden_size: int, scale_factor: int):
    if scale_factor == 1:
        upsample = nn.Identity()
    else:
        upsample = nn.Upsample(
            scale_factor=scale_factor,
            align_corners=True,
            mode="bilinear",
        )
    return nn.Sequential(
        nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
        nn.InstanceNorm2d(hidden_size),
        nn.ReLU(),
        upsample,
    )


class NeckDBNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up_2 = nn.Upsample(
            scale_factor=2,
            align_corners=True,
            mode="bilinear",
        )
        self.conv_up_8 = ConvUpsample(hidden_size, 8)
        self.conv_up_4 = ConvUpsample(hidden_size, 4)
        self.conv_up_2 = ConvUpsample(hidden_size, 2)
        self.conv_up_1 = ConvUpsample(hidden_size, 1)

    def forward(self, feature_maps: List):
        # features maps is list of feature map of scales:
        # - 1 / 4
        # - 1 / 8
        # - 1 / 16
        # - 1 / 32
        f_4, f_8, f_16, f_32 = feature_maps

        out_32 = self.conv_up_8(f_32)
        skip = self.up_2(f_32) + f_16
        out_16 = self.conv_up_4(skip)
        skip = self.up_2(skip) + f_8
        out_8 = self.conv_up_2(skip)
        skip = self.up_2(skip) + f_4
        out_4 = self.conv_up_1(skip)

        outputs = torch.cat([out_4, out_8, out_16, out_32], dim=1)
        return outputs
