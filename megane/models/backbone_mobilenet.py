from typing import List

from torch import nn


def ConvNormAct(in_channels, out_channels, kernel_size, **k):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        **k,
    )
    norm = nn.InstanceNorm2d(out_channels)
    act = nn.ReLU6(inplace=True)
    return nn.Sequential(conv, norm, act)


class MV2Block(nn.Module):
    """Mobilenet V2 block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride == 1:
            assert in_channels == out_channels
        middle_channels = in_channels * 4
        self.conv_1x1_a = ConvNormAct(in_channels, middle_channels, 1)
        self.conv_dw = ConvNormAct(
            middle_channels,
            middle_channels,
            kernel_size=3,
            groups=middle_channels,
            stride=stride,
        )
        self.conv_1x1_b = ConvNormAct(middle_channels, out_channels, 1)
        self.residual = stride == 1

    def forward(self, x):
        residual = x
        x = self.conv_1x1_a(x)
        x = self.conv_dw(x)
        x = self.conv_1x1_b(x)
        if self.residual:
            x = x + residual
        return x


def MV2Stage(in_channels, out_channels, num_layers):
    stage = nn.Sequential()
    strides = [1] * (num_layers - 1) + [2]
    in_channel_list = [in_channels] * num_layers
    out_channel_list = in_channel_list[1:] + [out_channels]
    for i in range(num_layers):
        layer = MV2Block(
            in_channel_list[i],
            out_channel_list[i],
            stride=strides[i],
        )
        stage.add_module(str(i), layer)
    return stage


class MobilenetV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        num_layers: List[int],
        project_size: int,
    ):
        super().__init__()
        assert len(in_channels) == len(num_layers) - 1
        self.in_conv = nn.Sequential(
            ConvNormAct(3, in_channels, 3, 2),
            ConvNormAct(in_channels, channels[0], 3, 2),
        )
        self.stages = nn.ModuleList()
        for c, n in zip(channels, num_layers):
            stage = MV2Stage(c, n)
            self.stages.append(stage)

    def forward(self, images):
        x = self.in_conv(images)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def mv2_t(project_size):
    return MobilenetV2(
        in_channels=16,
        channels=[32, 64, 96, 128, 192],
        num_layers=[3, 3, 3, 3],
        project_size=project_size,
    )
