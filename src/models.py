from typing import *

import torch
from torch.nn import functional as F
from torch import nn, Tensor, no_grad
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork

from .data import Sample
from . import utils


def _conv_norm_act(in_channels: int, out_channels: int, *a, **k):
    """Return a Conv Norm Activation Sequence

    Args:
        All the argument of `nn.Conv2d`

    Keywords:
        Conv:
            Type of convolution layer to use.
            Its initialization must be compatible with the one of `nn.Conv2d`.
            Default: `nn.Conv2d`
        All the keyword argument of `nn.Conv2d`.

    Returns:
        A `nn.Sequential` of Conv, `nn.InstanceNorm2d` and `nn.ReLU`
    """
    if "Conv" in k:
        Conv = k.pop("Conv")
    else:
        Conv = nn.Conv2d
    return nn.Sequential(
        Conv(in_channels, out_channels, *a, **k),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(),
    )


class UpscaleConcat(nn.Module):
    """Upscale each feature map and concat them with the next one,
    repeat until all feature maps are concatenated.

    Args:
        feature_size:
            Number of channels `C` of each feature returned by FPN.
        num_upscales:
            Number of features to be upscaled and concatenated.

    Inputs:
        features:
            An ordered dict mapping from name to features.
            The name does not matter.
            Each feature map has the shape [N, C, H(i), W(i)] and H(i+1) = 2 * H(i).

    Examples:
        from collections import OrderedDict
        import torch
        neck = UpscaleConcat(56, 4)
        features = {str(i): torch.rand(1, 56, 10 * i, 10 * i) for i in [1, 2, 4, 8]}
        features = OrderedDict(reversed(features.items())
        torch.Size([1, 224, 160, 160])
    """

    def __init__(self, feature_size: int, num_upscales: int):
        super().__init__()
        upscales = []
        for idx in range(num_upscales):
            channels = feature_size * (idx + 1)
            conv = _conv_norm_act(
                channels, channels, kernel_size=2, stride=2, Conv=nn.ConvTranspose2d
            )
            upscales.append(conv)
        self.upscales = nn.ModuleList(upscales)

    def forward(self, features: OrderedDict):
        output = None
        count = 0
        for k, feature in reversed(features.items()):
            upscale = self.upscales[count]
            count = count + 1
            if output is None:
                output = upscale(feature)
            else:
                output = torch.cat([output, feature], dim=1)
                output = upscale(output)
        return output


class BBSH(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()

        # Background and threshold
        self.bg_tr = nn.Conv2d(hidden_size, 2)
        # self.outputs = nn.Conv2d(


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.num_special_classes = 2
        self.aux_size = self.hidden_size // 4

        self.backbone = self._mk_backbone()
        self.fpn = self._mk_fpn()
        self.neck = UpscaleConcat(self.aux_size, 4)

        # Num class + 1 to compensate for background (no class)
        # self.head = PredictionHead(hidden_size, num_classes, num_special_classes)
        self.head = nn.Conv2d(
            self.hidden_size,
            self.num_classes + self.num_special_classes,
            kernel_size=(1, 3),
            padding=(0, 1),
        )

    def _mk_backbone(self):
        features = mobilenet_v3_large(num_classes=1).features
        layers = {str(layer): str(i) for i, layer in enumerate([3, 6, 9, 14])}
        backbone = IntermediateLayerGetter(features, layers)
        return backbone

    @no_grad()
    def _mk_fpn(self):
        image = torch.rand(1, 3, self.image_size, self.image_size)
        features = self.backbone(image)
        channels = [ft.shape[1] for ft in features.values()]
        fpn = FeaturePyramidNetwork(channels, self.hidden_size // len(channels))
        return fpn

    def forward(self, images: Tensor):
        features = self.backbone(images)
        features = self.fpn(features)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs

    def decode_sample(self, masks):
        pass

    def encode_sample(self, sample: Sample):
        """Mapping from plain sample to model domain

        Args:
            sample:
                Input `Sample` data.

        Returns:
            image:
                Torch image reprensentation of input image.
            target:
                Segmentation masks for training the model.
        """
        import numpy as np

        output_size = self.image_size // 2
        image = utils.prepare_input(sample.image, self.image_size, self.image_size)
        image = torch.FloatTensor(image)

        boxes = np.array(sample.boxes)
        shrink_boxes = utils.shrink(boxes)
        classes = np.array(sample.classes)

        # Object masks
        masks = []
        for c_idx in range(self.num_classes):
            id_mask = classes == c_idx
            mask = utils.draw_mask(
                output_size, output_size, shrink_boxes[id_mask], copy=False
            )
            masks.append(mask)

        # background mask
        bg_mask = 1 - utils.draw_mask(
            output_size, output_size, shrink_boxes, copy=False
        )
        border_mask = utils.draw_mask(output_size, output_size, boxes, copy=False) - (
            1 - bg_mask
        )

        masks.insert(0, bg_mask)
        masks.insert(1, border_mask)
        mask = np.stack(masks, axis=0)
        mask = torch.FloatTensor(mask)
        return image, mask

    def compute_loss(self, outputs, targets):
        """Compute prediction loss

        Args:
            outputs:
                Model outputs of shape [N, C, H, W]
            targets:
                Encoded targets of shape [N, C, H, W]
        """
        # Segmentation mask loss, consider this a localization loss
        l_loss = F.l1_loss(outputs, targets)

        # The classification loss
        # The loss is only calculated for each GT and each target labels
        # but not target labels against each others
        c_loss = F.cross_entropy(outputs, targets)

        loss = c_loss + l_loss
        return loss
