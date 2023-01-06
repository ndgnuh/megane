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
        if config.mode == "db":
            self.head = heads.DBHead(config.feature_size, 1)
            self.loss = losses.DBLoss()
        elif config.mode == "retina":
            self.head = heads.RetinaHead(config.feature_size)
            self.loss = losses.RetinaLoss()

    def forward(self, image):
        features = self.backbone(image)
        ic(image.shape, features.shape)
        return self.head(features)

    def training_step(self, batch):
        image, annotations = batch
        outputs = self(image)
        loss = self.compute_loss(outputs, annotations)

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.config.learning_rate)
        return optimizer
