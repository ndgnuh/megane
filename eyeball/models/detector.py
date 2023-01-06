from dataclasses import dataclass
from pytorch_lightning import LightningModule
from torch import nn, optim

from . import backbones, heads, losses
from ..tools import remember


@dataclass
class Config:
    backbone: str
    feature_size: int


class Detector(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = backbones.fpn_resnet18(config.feature_size)
        self.head = heads.DBHead(config.feature_size, 1)
        self.loss = losses.DBLoss()

    def forward(self, image):
        features = self.backbone(image)
        ic(image.shape, features.shape)
        return self.head(features)

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def configure_optimizer(self):
        optimizer = optim.AdamW(self, self.config.learning_rate)
        return optimizer
