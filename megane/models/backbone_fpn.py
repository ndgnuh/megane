from typing import List

import torch
from torch import nn
from torchvision import models

from megane.registry import backbones
from megane.models.backbone_vit import mobilevit18, mobilevit50
from megane.models.backbone_resnet import (
    resnet18,
    resnet34,
    resnet26,
    resnet50,
    tinyresnet26,
    tinyresnet50,
)


class HiddenLayerGetter(nn.Module):
    def __init__(self, model, positions):
        super().__init__()
        self.model = model
        self.masks = len(model) * [False]
        for i in positions:
            self.masks[i] = True

    @torch.no_grad()
    def get_out_channels(self):
        inputs = torch.rand(1, 3, 768, 768)
        return [output.shape[1] for output in self(inputs)]

    def forward(self, inputs) -> List:
        x = inputs
        outputs = []
        for keep, layer in zip(self.masks, self.model):
            x = layer(x)
            keep and outputs.append(x)
        return outputs


class FeaturePyramidNeck(nn.Module):
    def __init__(self, list_in_channels, out_channels):
        super().__init__()
        hidden_channels = out_channels // len(list_in_channels)
        self.upsample = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
        self.in_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
                for in_channels in list_in_channels
            ]
        )
        self.out_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, hidden_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Upsample(
                        scale_factor=2**idx, mode="bilinear", align_corners=True
                    ),
                )
                for idx, _ in enumerate(list_in_channels)
            ]
        )

        self.num_branches = len(list_in_channels)

    def forward(self, features: List):
        assert len(features) == self.num_branches
        # Input features
        features = [layer(ft) for layer, ft in zip(self.in_branch, features)]

        # Upscale combine
        outputs = [features[-1]]
        for i, ft in enumerate(reversed(features)):
            if i == 0:
                outputs.append(ft)
            else:
                output = self.upsample(outputs[-1])
                output = output + ft
                outputs.append(output)

        # Upscale concat
        features = [layer(ft) for layer, ft in zip(self.out_branch, reversed(outputs))]
        features = torch.cat(features, dim=1)
        return features


class FeaturePyramidNetwork(nn.Sequential):
    def __init__(self, net, imm_layers, out_channels: int):
        super().__init__()
        self.backbone = HiddenLayerGetter(net, imm_layers)
        self.fpn = FeaturePyramidNeck(self.backbone.get_out_channels(), out_channels)


@backbones.register()
def fpn_mobilenet_v3_large(out_channels: int):
    net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights).features
    return FeaturePyramidNetwork(net, [3, 6, 9, 16], out_channels)


@backbones.register()
def fpn_mobilenet_v3_small(out_channels):
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights).features
    model = FeaturePyramidNetwork(net, [1, 3, 8, 12], out_channels)
    return model


@backbones.register()
def fpn_resnet18(out_channels):
    net = resnet18()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_resnet26(out_channels):
    net = resnet26()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_resnet34(out_channels):
    net = resnet34()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_resnet50(out_channels):
    net = resnet50()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_tinyresnet26(out_channels):
    net = tinyresnet26()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_tinyresnet50(out_channels):
    net = tinyresnet50()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_mobilevit18(out_channels):
    net = mobilevit18()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model


@backbones.register()
def fpn_mobilevit50(out_channels):
    net = mobilevit50()
    model = FeaturePyramidNetwork(net, [1, 2, 3, 4], out_channels)
    return model
