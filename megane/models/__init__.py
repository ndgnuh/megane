from torch import Tensor, nn

from megane.models.api import ModelAPI
from megane.utils import init_from_ns


class backbones:
    from megane.models.backbone_fpn_resnet import (
        resnet_18,
        resnet_34,
        resnet_50,
        resnet_tiny_26,
        resnet_tiny_50,
    )
    from megane.models.backbone_mobilenet import afvit_t
    from megane.models.backbone_vit import mvit_11, mvit_18

    # from .backbone_fpn import FPNBackbone as fpn
    # from .backbone_fvit import FViTBackbone as fvit


class necks:
    from megane.models.neck_dbnet import NeckDBNet as dbnet
    from megane.models.neck_fpnconcat import FPNConcat as fpn_concat

    none = nn.Identity


class heads:
    from megane.models.head_dbnet import DBNet as dbnet
    from megane.models.head_dbgnet import DBNet as dbgnet


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size

        # Backbone
        self.backbone = init_from_ns(backbones, config.backbone)
        self.neck = init_from_ns(necks, config.neck)
        try:
            self.head = init_from_ns(heads, config.head, config)
        except Exception:
            self.head = init_from_ns(heads, config.head)

        # Assign configs to allow access
        self.head.config = config

        # Ensure head is an ModelAPI compat
        assert isinstance(self.head, ModelAPI)

        # Delegation
        self.encode_sample = self.head.encode_sample
        self.decode_sample = self.head.decode_sample
        self.compute_loss = self.head.compute_loss
        self.collate_fn = self.head.collate_fn
        self.visualize_outputs = self.head.visualize_outputs
        self.set_infer(False)

    def forward(self, images: Tensor, targets: Tensor | None = None):
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
