from pytorch_lightning.lite import LightningLite
from . import models


class Trainer(LightningLite):
    def __init__(self, config):
