from dataclasses import dataclass
from pytorch_lightning import LightningModule
from torch import nn, optim

from . import backbones, heads, losses
from ..tools import remember


@dataclass
class Config:
    backbone: str
    mode: str
    feature_size: int
    learning_rate: float = 1e-3


class Detector(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = config.mode

        Backbone = getattr(backbones, config.backbone)
        self.backbone = Backbone(config.feature_size)
        self.head = heads.HeadMixin(config.mode, config.head_options)

    def forward(self, image):
        features = self.backbone(image)
        return self.head(features)
