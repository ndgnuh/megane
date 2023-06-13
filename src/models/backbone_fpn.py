from collections import OrderedDict

import torch
from torch import nn, no_grad
from torchvision import models as vision_models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork

from ..configs import ModelConfig


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
            conv = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2,
                                   stride=2, groups=channels),
                nn.InstanceNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(),
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


class FPNBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_layers = len(config.backbone.layers)
        self.config = config
        self.aux_size = config.hidden_size // self.num_layers
        self.image_size = config.image_size
        assert self.image_size % 32 == 0

        # Feature extractor
        self.backbone = self._mk_backbone()

        # FPN
        self.fpn = self._mk_fpn()

    def _mk_backbone(self):
        arch = self.config.backbone.arch
        feature_module = self.config.backbone.feature_module
        layers = self.config.backbone.layers

        features = getattr(vision_models, arch)(num_classes=1)
        if feature_module is not None:
            features = getattr(features, feature_module)
        layers = {str(layer): str(i) for i, layer in enumerate(layers)}
        backbone = IntermediateLayerGetter(features, layers)
        return backbone

    @no_grad()
    def _mk_fpn(self):
        image_size = self.config.image_size
        hidden_size = self.config.hidden_size

        image = torch.rand(1, 3, image_size, image_size)
        features = self.backbone(image)
        channels = [ft.shape[1] for ft in features.values()]
        fpn = FeaturePyramidNetwork(channels, hidden_size // len(channels))
        return fpn

    def forward(self, images):
        features = self.backbone(images)
        features = self.fpn(features)
        return features
