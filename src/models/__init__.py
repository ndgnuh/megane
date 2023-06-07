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
from ..configs import ModelConfig, FViTConfig, FPNConfig
from .head_dbbn import DBBNHead
from .backbone_fpn import FPNBackbone
from .backbone_fvit import FViTBackbone


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        if isinstance(config.backbone, FViTConfig):
            self.backbone = FViTBackbone(config)
        elif isinstance(config.backbone, FPNConfig):
            self.backbone = FPNBackbone(config)
        else:
            raise ValueError(f"Unsupported backbone {config.backbone}")
        self.head = DBBNHead(config)

        # Delegation
        self.encode_sample = self.head.encode_sample
        self.decode_sample = self.head.decode_sample
        self.compute_loss = self.head.compute_loss
        self.set_infer(False)

    def forward(self, images: Tensor):
        features = self.backbone(images)
        outputs = self.head(features)
        return outputs

    def set_infer(self, infer: bool):
        """
        Set inference mode.
        This is different from eval() in that eval() can be used when validating
        but this cannot.

        Inference mode remove some of the auxiliary channels of the model.
        Setting infer to True also calls `eval()`, while setting it to False call `train()`.
        """
        self.infer = infer
        if infer:
            self.eval()
        else:
            self.train()
        for module in self.modules():
            assert not hasattr(module, "infer") or isinstance(getattr(module, "infer"), bool)
            module.infer = infer
