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
                nn.Conv2d(in_channel, out_channel, 3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
            for in_channel in in_channels
        ])
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.out_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, mid_channel, 3, padding=1),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),
                nn.Upsample(scale_factor=2**i,
                            mode="bilinear",
                            align_corners=True)
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


def fpn_resnet34(output_size: int):
    cnn = models.resnet18()
    return FPNBackbone(cnn, output_size, ["layer1", "layer2", "layer3", "layer4"])


def fpn_resnet50(output_size: int):
    cnn = models.resnet18()
    return FPNBackbone(cnn, output_size, ["layer1", "layer2", "layer3", "layer4"])


def fpn_shufflenet_v2_x0_5(output_size: int):
    cnn = models.shufflenet_v2_x0_5()
    layers = ['maxpool', 'stage2', 'stage3', 'conv5']
    return FPNBackbone(cnn, output_size, layers)


def fpn_mobilenet_v2(output_size: int):
    cnn = models.mobilenet_v2().features
    return FPNBackbone(cnn, output_size, ['3', '6', '12', '16'])


def fpn_mobilenet_v3_large(output_size: int):
    cnn = models.mobilenet_v3_large().features
    return FPNBackbone(cnn, output_size, ['3', '6', '12', '16'])


def fpn_mobilenet_v3_small(output_size: int):
    cnn = models.mobilenet_v3_small().features
    return FPNBackbone(cnn, output_size,  ['1', '3', '8', '12'])


def fpn_efficientnet_b3(output_size: int):
    cnn = models.efficientnet_b3().features
    return FPNBackbone(cnn, output_size, ['2', '3', '5', '7'])
