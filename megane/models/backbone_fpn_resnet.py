from typing import List

from torch import nn
from megane.registry import backbones


def ConvNorm(*a, **k):
    # There is no point adding bias
    # we are just going to normalize right after that
    conv = nn.Conv2d(*a, **k, bias=False)
    norm = nn.InstanceNorm2d(conv.out_channels)
    relu = nn.ReLU()
    return nn.Sequential(conv, norm, relu)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNorm(in_channels, out_channels, 3, stride=stride, padding=1),
            ConvNorm(out_channels, out_channels, 3, padding=1),
        )

        if stride > 1 or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x) + self.residual(x)
        out = self.relu(out)
        return out


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNorm(in_channels, in_channels, 1),
            ConvNorm(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                groups=in_channels,
            ),
            ConvNorm(in_channels, out_channels, 1),
        )

        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return out


def ResNetStage(
    Block, hidden_size: int, output_size: int, num_layers: int, stride: int = 1
):
    layers = [Block(hidden_size, output_size, stride=stride)]
    for _ in range(num_layers - 1):
        layer = ResidualBlock(output_size, output_size)
        layers.append(layer)
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        num_layers: List[int],
        block_type: ResidualBlock | BottleNeckBlock,
        project_size: int = None,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, hidden_sizes[0], 7, padding=3, stride=2),
            nn.InstanceNorm2d(hidden_sizes[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()

        strides = [1, 2, 2, 2]
        for i, input_size in enumerate(hidden_sizes):
            # Try to get the next layer size, if can't use the current one
            try:
                output_size = hidden_sizes[i + 1]
            except Exception:
                output_size = input_size

            # Stage layer
            num_layer = num_layers[i]
            stride = strides[i]
            stage = ResNetStage(
                block_type,
                input_size,
                output_size,
                num_layer,
                stride,
            )

            # Projection layer
            if project_size is None:
                projection = nn.Identity()
            else:
                projection = nn.Conv2d(output_size, project_size, 1, bias=False)

            # Add layers
            self.stages.append(stage)
            self.projections.append(projection)

    def forward(self, images):
        x = self.input(images)
        outputs = []
        for stage, project in zip(self.stages, self.projections):
            x = stage(x)
            outputs.append(project(x))
        return outputs


@backbones.register()
def resnet_18(project_size: int | None = None):
    hidden_sizes = [64, 128, 256, 512]
    num_layers = [2, 2, 2, 2]
    return ResNet(hidden_sizes, num_layers, ResidualBlock, project_size)


@backbones.register()
def resnet_34(project_size: int | None = None):
    hidden_sizes = [64, 128, 256, 512]
    num_layers = [3, 4, 6, 3]
    return ResNet(hidden_sizes, num_layers, ResidualBlock, project_size)


@backbones.register()
def resnet_50(project_size: int | None = None):
    hidden_sizes = [64, 128, 256, 512, 2048]
    num_layers = [3, 4, 6, 3]
    return ResNet(hidden_sizes, num_layers, BottleNeckBlock, project_size)


@backbones.register()
def resnet_tiny_26(project_size: int | None = None):
    hidden_sizes = [48, 96, 128, 256]
    num_layers = [2, 2, 2, 2]
    return ResNet(hidden_sizes, num_layers, BottleNeckBlock, project_size)


@backbones.register()
def resnet_tiny_50(project_size: int | None = None):
    hidden_sizes = [48, 96, 128, 256]
    num_layers = [3, 4, 6, 3]
    return ResNet(hidden_sizes, num_layers, BottleNeckBlock, project_size)
