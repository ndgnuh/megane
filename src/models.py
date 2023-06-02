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


class BgThreshTextNoise(nn.Module):
    """Very specific prediction head for text detection.

    Predict toggle-able channels of text, noise, background and thresholds.
    To use this head, prepare the data so that the text class is 0 and the noise class is 1.
    Otherwise, specify the class index in the arguments.

    The idea is to separate the text from the background using a classification loss.
    So the loss is computed as crossentropy of class labels, using merged feature maps:
    - background
    - text-border
    - text
    - noise
    Since noise can overlap with texts, their ground truth are encoded so that the text mask is always priortized.
    During inference, only the text channel is computed.

    Args:
        hidden_size:
            The feature channel dimensions
        image_size:
            The input image size, this is required to encode the label
        text_class:
            Class index for text
        noise_class:
            Class index for noise

    Inputs:
        features:
            Tensor of shape [N, C, H, W]

    Methods:
        - encode_sample
        - decode_sample
        - compute_loss
    """

    def __init__(
        self,
        hidden_size: int,
        image_size: int,
        text_class: int = 0,
        noise_class: int = 1,
    ):
        super().__init__()
        self.text_class = text_class
        self.noise_class = noise_class
        self.conv_text_class = text_class + 2
        self.conv_noise_class = noise_class + 2
        self.image_size = image_size

        # Background and threshold
        self.heads = nn.ModuleList([nn.Conv2d(hidden_size, 1, 1) for _ in range(4)])

    def forward(self, features, return_all=False):
        if self.training or return_all:
            outputs = torch.cat([conv(features) for conv in self.heads], dim=1)
        else:
            outputs = self.heads[self.conv_text_class](features)
        return outputs

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
        shrink, expand = utils.shrink_expand(boxes, r=0.4)
        classes = np.array(sample.classes)

        # Mask draw helper
        def dmask(boxes, c_idx, mode="max"):
            mask = utils.draw_mask(
                output_size, output_size, boxes[classes == c_idx], copy=False, mode=mode
            )
            mask = mask.astype("bool")
            return mask

        # Draw masks
        noise_mask = dmask(boxes, self.noise_class, "round")
        thres_mask = dmask(expand, self.text_class, "max")
        text_mask = dmask(shrink, self.text_class, "max")
        bg_mask = np.ones([output_size, output_size], dtype=bool)

        # Target masks
        thres_mask = thres_mask & (~text_mask)
        bg_mask = bg_mask & (~text_mask) & (~noise_mask) & (~text_mask)
        masks = np.stack((bg_mask, thres_mask, text_mask, noise_mask), axis=0)

        return image, masks

    def compute_loss(self, outputs, targets):
        f_targets = targets * 1.0
        # Basic segmentation loss
        loss = F.l1_loss(outputs, f_targets)
        loss = F.mse_loss(outputs, f_targets) + loss
        loss = loss / 2

        # Loss for bg, text and text threshold
        idx = [0, 1, 2]
        c_loss_t = F.cross_entropy(outputs[:, idx], f_targets[:, idx] * 1.0)

        # Loss for bg and noise
        idx = [0, 3]
        c_loss_n = F.cross_entropy(outputs[:, idx], f_targets[:, idx] * 1.0)

        # Loss for text and noise
        gt_bg, gt_tt, gt_tp, gt_np = targets.chunk(4, dim=1)
        gt_np = gt_np & (~gt_tt) & (~gt_tp)
        targets_tn = torch.cat([gt_tt, gt_tp, gt_np], dim=1) * 1.0
        c_loss_tn = F.cross_entropy(outputs[:, [1, 2, 3]], targets_tn)

        loss = loss + c_loss_t + c_loss_n + c_loss_tn
        return loss


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
        self.head = BgThreshTextNoise(self.hidden_size, self.image_size)
        self.encode_sample = self.head.encode_sample
        self.compute_loss = self.head.compute_loss

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
