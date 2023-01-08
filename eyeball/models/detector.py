from dataclasses import dataclass
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch

from . import backbones, heads, losses
from .. import processor
from ..tools import init
from ..tools import remember


class Detector(nn.Sequential):
    def __init__(self, config):
        super().__init__()
        self.backbone = init.init_from_ns(
            backbones,
            config['backbone'],
            config['backbone_options']
        )
        self.head = init.init_from_ns(
            heads,
            config['head'],
            config['head_options']
        )

        self.processor = init.init_from_ns(
            processor,
            config['processor'],
            config.get('processor_options', {})
        )

        if 'weights' in config:
            w = torch.load(config['weights'], map_location='cpu')
            self.model.load_state_dict(w)
