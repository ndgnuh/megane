from torch import nn

from megane.registry import heads


class ResidualRefinementModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        num_layers = 4
        self.in_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.in_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.MaxPool2d(2),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear"),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.out_2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        residuals = [x]
        for layer in self.in_branch:
            x = layer(x)
            residuals.insert(0, x)

        for i, layer in enumerate(self.out_branch):
            x = layer(x + residuals[i])

        x = self.out_1(x + residuals[-1])
        x = self.out_2(x)

        return x


class HeadWithRRM(nn.Module):
    def __init__(self, head, rrm):
        super().__init__()
        self.head = head
        self.rrm = rrm

    def forward(self, x):
        x = self.head(x)
        dx = self.rrm(x)
        return x + dx
