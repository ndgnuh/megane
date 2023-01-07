from torch import nn


class RetinaRegress(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_anchors: int,
        box_size: int = 4,
        num_convs: int = 4
    ):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(num_convs):
            pass


class RetinaHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        feature_size: int = 256,
        num_classes: int = 1,
        box_size: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Bounding box regression head
        self.regression = nn.Sequential()
        for i in range(4):
            in_channel = in_channels if i == 1 else feature_size
            self.regression.add_module(f"step_{i}", nn.Sequential(
                nn.Conv2d(in_channel, feature_size, 3, padding=1),
                nn.GELU(approximate='tanh')
            ))
        self.regression.add_module("output", nn.Conv2d(
            feature_size, num_anchors * box_size, 3, padding=1
        ))

    def forward(self, features):
        # c h w -> c (h w) -> (h w) c
        boxes = self.regression(features)
        boxes = boxes.flatten(-2).transpose(-1, -2)

        if self.num_classes > 1:
            classes = self.classification(features)
        else:
            classes = None
        return boxes, classes


class DBHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        num_upscales: int = 2,
        activation: str = 'ReLU'
    ):
        super().__init__()
        self.num_upscales = num_upscales
        self.num_classes = num_classes
        self.input_size = input_size
        self.activation = activation
        assert hasattr(nn, activation), f"Invalid activation {activation}"

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
        input_size = self.input_size
        head_size_d4 = self.input_size // 4

        layers = [
            nn.Conv2d(input_size, head_size_d4, 3, padding=1, bias=False),
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
            act = getattr(nn, self.activation)()
            layers.extend([conv, act])

        return nn.Sequential(*layers)


def HeadMixin(mode, options):
    if mode == "db":
        return DBHead(**options)
    if mode == "retina":
        return RetinaHead(**options)
    raise ValueError("Unsupported mode")
