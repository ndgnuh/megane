from torch import nn


class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand, stride):
        super().__init__()
        mid_channels = in_channels * expand
        self.skip = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                groups=mid_channels,
                stride=stride,
                bias=False,
            ),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(mid_channels),
        )

    def forward(self, img):
        if self.skip:
            return img + self.conv(img)
        else:
            return self.conv(img)


class MobileNetV2(nn.Module):
    def default_config(self):
        # c, t, s, n
        return [
            [16, 24, 32, 64, 96, 160, 320],
            [1, 6, 6, 6, 6, 6, 6],
            [1, 2, 2, 2, 1, 2, 1],
            [1, 2, 3, 4, 3, 3, 1],
        ]

    def __init__(self, config=None, stem_size: int = 32, output_size: int = 1280):
        super().__init__()
        if config is None:
            config = self.default_config()

        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(3, stem_size, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(stem_size),
            nn.ReLU6(True),
        )

        # MV2 blocks
        pc = stem_size
        layers = [stem]
        for c, t, s, n in zip(*config):
            layer = MV2Block(pc, c, expand=t, stride=s)
            layers.append(layer)
            for i in range(1, n):
                layer = MV2Block(c, c, expand=t, stride=1)
                layers.append(layer)
            pc = c

        # Final conv
        out_conv = nn.Sequential(
            nn.Conv2d(pc, output_size, 1, bias=False),
            nn.InstanceNorm2d(output_size),
            nn.ReLU6(True),
        )
        layers.append(out_conv)

        # Features
        self.features = nn.ModuleList(layers)

        # Output keeping masks
        self.keep = [False] * len(self.features)
        for i in (3, 6, 13, 18):
            self.keep[i] = True

    def forward(self, image):
        outputs = []
        for keep, layer in zip(self.keep, self.features):
            image = layer(image)
            keep and outputs.append(image)

        # Output sizes, with input = 224 * 224
        # torch.Size([1, 24, 56, 56])
        # torch.Size([1, 32, 28, 28])
        # torch.Size([1, 96, 14, 14])
        # torch.Size([1, 1280, 7, 7])
        return outputs
