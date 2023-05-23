from dataclasses import dataclass
from functools import cached_property

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


@dataclass
class MeganeConfig:
    num_classes: int
    num_box_dims: int = 8
    scheme: str = "detr"


class MeganeDetector(nn.Module):
    def __init__(self, config):
        assert config.scheme == "detr", "Other scheme is not supported yet"
        super().__init__()
        self.feature = mobilenet_v3_small(num_classes=1).features
        self.num_classes = config.num_classes + 1  # background
        self.num_box_dims = config.num_box_dims

        self.classify_head = nn.Conv2d(
            self.num_feature_channels, self.num_classes, 1, bias=False
        )
        self.localize_head = nn.Conv2d(
            self.num_feature_channels, self.num_box_dims, 1, bias=False
        )

    @cached_property
    def num_feature_channels(self):
        with torch.no_grad():
            image = torch.rand(1, 3, 256, 256)
            feature_channels = self.feature(image).shape[1]
        return feature_channels

    def forward(self, images):
        # extract features
        feature = self.feature(images)
        # predictions
        class_logits = self.classify_head(feature)
        boxes = self.localize_head(feature)

        # b c h w -> b c (h w) -> b (h w) c
        class_logits = class_logits.flatten(-2).transpose(-1, -2)
        boxes = boxes.flatten(-2).transpose(-1, -2)
        return class_logits, boxes
