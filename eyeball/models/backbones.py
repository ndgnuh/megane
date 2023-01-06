from torch import nn
from typing import List
from math import floor
import torch
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import models


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channel: int,
    ):
        super().__init__()
        num_inputs = len(in_channels)
        mid_branch_size = int(floor(out_channel / num_inputs))
        mid_residual_size = out_channel - (num_inputs - 1) * mid_branch_size
        mid_channels = [mid_residual_size] + \
            [mid_branch_size] * (num_inputs - 1)

        self.in_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.GELU(approximate='tanh')
            )
            for in_channel in in_channels
        ])

        self.out_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, mid_channel, 3, padding=1),
                nn.GELU(approximate='tanh'),
                nn.UpsamplingBilinear2d(scale_factor=2**i)
            )
            for (i, mid_channel) in enumerate(mid_channels)
        ])

    def forward(self, features):
        features = [layer(f) for layer, f in zip(self.in_branch, features)]
        features = [layer(f) for layer, f in zip(self.out_branch, features)]
        features = torch.cat(tuple(features), dim=-3)
        return features


class FPNBackbone(nn.Module):
    def __init__(self, cnn: nn.Module, out_channel: int, layers: List[str]):
        super().__init__()
        imm_layers = {layer: str(i) for i, layer in enumerate(layers)}
        self.cnn = IntermediateLayerGetter(cnn, imm_layers)

        with torch.no_grad():
            input_size = 512
            x = torch.rand(1, 3, input_size, input_size)
            features = self.cnn(x)
            in_channels = [f.shape[-3] for f in features.values()]

        self.fpn = FPN(in_channels, out_channel)

    def forward(self, image):
        features = self.cnn(image)
        features = self.fpn(features.values())
        return features


def fpn_resnet18(output_size: int):
    cnn = models.resnet18()
    return FPNBackbone(cnn, output_size, ["layer1", "layer2", "layer3", "layer4"])
