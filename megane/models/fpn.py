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
                nn.InstanceNorm2d(out_channel),
                nn.ReLU()
            )
            for in_channel in in_channels
        ])
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.out_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, mid_channel, 3, padding=1),
                nn.InstanceNorm2d(mid_channel),
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
        return features


class FPNBackbone(nn.Module):
    def __init__(self, preset: str, out_channel: int):
        super().__init__()
        fpn_config = presets[preset]
        feature_path = fpn_config.get('feature_path', None)
        cnn = getattr(models, preset)(num_classes=1)
        if feature_path is not None:
            cnn = getattr(cnn, feature_path)
        imm_layers = {layer: str(i)
                      for i, layer in enumerate(fpn_config['layers'])}
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


presets = {
    "resnet18": dict(
        layers=["layer1", "layer2", "layer3", "layer4"],
        feature_path=None
    ),
    "resnet34": dict(
        layers=["layer1", "layer2", "layer3", "layer4"],
        feature_path=None
    ),
    "resnet50": dict(
        layers=["layer1", "layer2", "layer3", "layer4"],
        feature_path=None
    ),
    "shufflenet_v2_x0_5": dict(
        layers=["maxpool", "stage2", "stage3", "conv5"],
        feature_path=None
    ),
    "mobilenet_v2": dict(
        layers=["3", "6", "12", "16"],
        feature_path="features"
    ),
    "mobilenet_v3_large": dict(
        layers=["3", "6", "12", "16"],
        feature_path="features"
    ),
    "mobilenet_v3_small": dict(
        layers=["1", "3", "8", "12"],
        feature_path="features"
    ),
    "efficientnet_b3": dict(
        layers=["2", "3", "5", "7"],
        feature_path="features"
    ),
}
