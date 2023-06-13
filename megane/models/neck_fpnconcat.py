from typing import List

import torch
from torch import nn


class UpsampleConv(nn.Module):
    def __init__(self, hidden_size: int, num_upscales: int):
        super().__init__()
        layers = []
        for i in range(num_upscales):
            layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
                nn.InstanceNorm2d(hidden_size),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, 1),
                nn.InstanceNorm2d(hidden_size),
                nn.ReLU(),
            )
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FPNConcat(nn.Module):
    def __init__(self, hidden_size: int, num_upsamples: List[int]):
        super().__init__()
        mid_size = hidden_size // len(num_upsamples)
        self.layers = nn.ModuleList(
            [UpsampleConv(mid_size, num_upsample) for num_upsample in num_upsamples]
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 1), padding=(1, 0)),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1)),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(),
        )

    def forward(self, features):
        outputs = []
        count = 0
        for v in features:
            layer = self.layers[count]
            output = layer(v)
            outputs.append(output)
            count += 1
        outputs = torch.cat(outputs, dim=1)
        outputs = self.output(outputs)
        return outputs
