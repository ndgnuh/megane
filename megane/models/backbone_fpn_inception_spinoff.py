import torch
from torch import nn


def ConvIR(in_channels, out_channels, *a, **k):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *a, **k),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(),
    )


class Block(nn.Module):
    def __init__(self, channels, out_channels, stride=1):
        super().__init__()
        hidden_channels = channels // 4
        self.skip = stride == 1 and out_channels == channels
        self.conv_1 = ConvIR(
            channels, hidden_channels, (3, 1), padding=(1, 0), stride=stride
        )
        self.conv_2 = ConvIR(
            channels, hidden_channels, (1, 3), padding=(0, 1), stride=stride
        )
        self.conv_3 = ConvIR(channels, hidden_channels, 3, padding=1, stride=stride)
        self.conv_4 = ConvIR(channels, hidden_channels, 1, stride=stride)
        self.mixer = ConvIR(channels, out_channels, 1)

    def forward(self, x):
        outputs = [
            self.conv_1(x),
            self.conv_2(x),
            self.conv_3(x),
            self.conv_4(x),
        ]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.mixer(outputs)
        if self.skip:
            outputs = x + outputs
        return outputs


class Stage(nn.Module):
    def __init__(self, hidden_size, out_channels, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == (num_blocks - 1):
                stride = 2
                h1 = hidden_size
                h2 = out_channels
            else:
                stride = 1
                h1 = h2 = hidden_size
            block = Block(h1, h2, stride=stride)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Network(nn.Module):
    def __init__(self, num_blocks, hidden_sizes):
        super().__init__()
        stages = []

        stage1 = nn.Sequential(
            ConvIR(3, hidden_sizes[0], 7, stride=2, padding=3),
            ConvIR(hidden_sizes[0], hidden_sizes[0], 5, stride=2, padding=2),
        )
        stages.append(stage1)

        for i, num_blocks_ in enumerate(num_blocks):
            stage = Stage(hidden_sizes[i], hidden_sizes[i + 1], num_blocks_)
            stages.append(stage)

        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(
            [
                ConvIR(h1, hidden_sizes[-1] // len(hidden_sizes), 1)
                for h1 in hidden_sizes
            ]
        )

    def forward(self, images):
        outputs = []
        for stage, project in zip(self.stages, self.projections):
            images = stage(images)
            output = project(images)
            outputs.append(output)
        return outputs
