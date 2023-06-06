from typing import *

import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, Tensor, no_grad
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import functional as TF

from ..data import Sample
from .. import utils
from ..configs import ModelConfig
from .head_dbbn import DBBNHead
from .backbone_fpn import FPNBackbone


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        self.backbone = FPNBackbone(config)
        self.head = DBBNHead(config)

        # Delegation
        self.encode_sample = self.head.encode_sample
        self.decode_sample = self.head.decode_sample
        self.compute_loss = self.head.compute_loss

    def forward(self, images: Tensor):
        features = self.backbone(images)
        outputs = self.head(features)
        return outputs
