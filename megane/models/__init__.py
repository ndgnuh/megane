import torch
from torch import Tensor, nn
from torchvision import transforms as T

from megane.models import (
    backbone_vit,
    backbone_mobilenet,
    backbone_resnet,
    backbone_fpn,
    head_dbnet,
)
from megane.models.api import ModelAPI
from megane.utils import init_from_ns
from megane.registry import backbones, heads


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Backbone
        self.preprocess = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.backbone = init_from_ns(backbones, config.backbone)
        self.head = init_from_ns(heads, config.head)

        # Delegation
        self.encode_sample = self.head.encode_sample
        self.decode_sample = self.head.decode_sample
        self.compute_loss = self.head.compute_loss
        self.collate_fn = self.head.collate_fn
        self.visualize_outputs = self.head.visualize_outputs
        self.set_infer(False)

    def forward(self, images: Tensor, targets: Tensor | None = None):
        images = self.preprocess(images)
        features = self.backbone(images)
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
