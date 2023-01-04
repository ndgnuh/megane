from torch import nn


class RetinaHead(nn.Module):
    pass


class DBHead(nn.Module):
    def __init__(
        self,
        head_size: int,
        num_classes: int = 1,
        num_upscales: int = 4
    ):
        super().__init__()
        self.num_upscales = num_upscales
        self.num_classes = num_classes
        self.head_size = head_size
        self.prob_head = self.make_branch()
        self.thres_head = self.make_branch()

    def forward(self, features):
        if self.training:
            thres = self.thres_head(features)
        else:
            thres = None
        prob = self.prob_head(features)
        return prob, thres

    def make_branch(self):
        head_size = self.head_size
        head_size_d4 = self.head_size // 4

        layers = [
            nn.Conv2d(head_size, head_size_d4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_size_d4),
            nn.ReLU(),
        ]

        for i in range(self.num_upscales):
            if i == self.num_upscales - 1:
                output_size = self.num_classes
            else:
                output_size = head_size_d4

            conv = nn.ConvTranspose2d(
                head_size_d4,
                output_size,
                kernel_size=2,
                stride=2,
                bias=False
            )
            act = nn.ReLU()
            layers.extend([conv, act])

        return nn.Sequential(*layers)
