from torch import nn
import torch

from .db import DBHead
from .fpn import FPNBackbone


class DBNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        hidden_size: int,
        num_classes: int = 1,
        weights: str = None,
    ):
        super().__init__()
        self.fpn = FPNBackbone(backbone, hidden_size)
        self.head = DBHead(hidden_size, num_classes=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights, map_location=True))

    def forward(self, image):
        features = self.fpn(image)
        features = torch.cat(features, dim=-3)
        proba_maps, threshold_maps = self.head(features)
        return proba_maps, threshold_maps

    @classmethod
    def from_config(cls, config):
        return cls(hidden_size=config['hidden_size'],
                   backbone=config['backbone'],
                   num_classes=config['num_classes'],
                   weights=config.get("weights", None))
