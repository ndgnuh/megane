from torch import nn
import torch

from .db import DBHead
from .fpn import FPNBackbone


class DBNet(nn.Module):
    def __init__(self, fpn_preset: str, hidden_size: int):
        super().__init__()
        self.fpn = FPNBackbone(fpn_preset, hidden_size)
        self.head = DBHead(hidden_size)

    def forward(self, image):
        features = self.fpn(image)
        features = torch.cat(features, dim=-3)
        proba_maps, threshold_maps = self.head(features)
        return proba_maps, threshold_maps
