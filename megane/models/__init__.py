from typing import *

import numpy as np
import torch
from torch import Tensor, nn, no_grad
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import functional as TF

from .. import configs, utils
from ..configs import (
    FPNConfig,
    FViTConfig,
    HeadConfig,
    ModelConfig,
    PCViTConfig,
    Seq2seqConfig,
)
from ..data import Sample
from .api import ModelAPI
from .backbone_fpn import FPNBackbone
from .backbone_fvit import FViTBackbone
from .backbone_pcvit import PCViTBackbone
from .head_dbbn import DBBNHead
from .head_seq2seq import Seq2seq


class backbones:
    from .backbone_fpn_inception_spinoff import Network as fpn_spin

    # from .backbone_fpn import FPNBackbone as fpn
    # from .backbone_fvit import FViTBackbone as fvit


class necks:
    from .neck_fpnconcat import FPNConcat as fpn_concat

    none = nn.Identity


class heads:
    from .head_segm import SegmentHead as segment


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size

        # Backbone
        self.backbone = utils.init_from_ns(backbones, config.backbone)
        self.neck = utils.init_from_ns(necks, config.neck)
        self.head = utils.init_from_ns(heads, config.head)

        # Ensure head is an ModelAPI compat
        assert isinstance(self.head, ModelAPI)

        # Delegation
        self.encode_sample = self.head.encode_sample
        self.decode_sample = self.head.decode_sample
        self.compute_loss = self.head.compute_loss
        self.collate_fn = self.head.collate_fn
        self.set_infer(False)

    def forward(self, images: Tensor, targets: Optional[Tensor] = None):
        features = self.backbone(images)
        features = self.neck(features)
        outputs = self.head(features, targets)
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
            assert not hasattr(module, "infer") or isinstance(
                getattr(module, "infer"), bool
            )
            module.infer = infer
