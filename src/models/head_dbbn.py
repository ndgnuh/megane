from typing import *

import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torch import nn


from .. import utils
from ..data import Sample
from ..configs import ModelConfig
from .losses import ContourLoss


def visualize_outputs(outputs: torch.Tensor):
    """Create preview for logger

    Args:
        outputs:
            Tensor of shape [B, C, H, W]

    Returns:
        image:
            Tensor of shape [1, B * H, C * W]
    """
    if outputs.ndim == 3:
        outputs = outputs.unsqueeze(0)
    B, C, H, W = outputs.shape

    # This actually gives much better results than sigmoid, softmax or thresholding
    images = torch.clip(torch.tanh(outputs), 0, 1)

    # Stack batches to height
    images = torch.cat([image for image in images], dim=-2)

    # Stack channels to width
    images = torch.cat([image for image in images], dim=-1)

    # To [C, H, W]
    images = images.unsqueeze(0)
    return images


class DBBNHead(nn.Module):
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

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.text_class = 0
        self.noise_class = 1
        self.conv_text_class = self.text_class + 2
        self.conv_noise_class = self.noise_class + 2
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.final_div_factor = config.head.final_div_factor
        assert self.image_size % self.final_div_factor == 0

        # Background and threshold
        self.heads = nn.ModuleList(
            [nn.Conv2d(self.hidden_size, 1, 1) for _ in range(4)]
        )
        self.inferring = False
        self.visualize_outputs = visualize_outputs
        self.contour_loss = ContourLoss(kernel_size=config.head.contour_loss_kernel)

    def forward(self, features, returns_all=False):
        if self.inferring:
            outputs = self.heads[self.conv_text_class](features)
        else:
            outputs = torch.cat([conv(features) for conv in self.heads], dim=1)
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

        output_size = self.image_size // self.final_div_factor
        image = utils.prepare_input(sample.image, self.image_size, self.image_size)
        image = torch.FloatTensor(image)

        # Compute shrink/expand distance
        boxes = sample.boxes
        areas = list(map(utils.polygon.polygon_area, boxes))
        lengths = list(map(utils.polygon.polygon_perimeter, boxes))
        dists = [A * (1 - 0.4 ** 2) / L for (A, L) in zip(areas, lengths)]

        # Generate shrink/expanded polygons
        shrink = [utils.offset_polygon(p, -d) for (p, d) in zip(boxes, dists)]
        expand = [utils.offset_polygon(p, d) for (p, d) in zip(boxes, dists)]

        # Mask draw helper
        classes = sample.classes
        def dmask(boxes, c_idx):
            boxes_by_class = [boxes[i] for (i, c) in enumerate(classes) if c == c_idx]
            mask = utils.draw_mask_v2(output_size, output_size, boxes_by_class)
            mask = mask.astype("bool")
            return mask

        # Draw masks
        noise_mask = dmask(boxes, self.noise_class)
        thres_mask = dmask(expand, self.text_class)
        text_mask = dmask(shrink, self.text_class)

        # Target masks
        thres_mask = thres_mask & (~text_mask)
        masks = np.stack((thres_mask, text_mask, noise_mask), axis=0)

        return image, masks

    def decode_sample(self, inputs, outputs, sample: Optional[Sample] = None) -> Sample:
        """Return a sample from raw model outputs
        Args:
            inputs:
                Model encoded input image of shape [3, H, W]
            outputs:
                Model raw outputs of shape [C, H, W].
                If C = 1, the first channel will be used.
                If C = 4, the third channel will be used.
                If C = 3 (ground truth), the second channel will be used.
            sample:
                The input sample, this is only used to get the image.

        Returns:
            The decoded sample.
        """
        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu()
        if sample is None:
            image = TF.to_pil_image(inputs)
            h, w = inputs.shape[-2:]
        else:
            image = sample.image
            w, h = image.size
        C = outputs.shape[0]
        if C == 1:
            text_mask = outputs[0]
        elif C == 4:
            text_mask = outputs[2]
        elif C == 3:
            text_mask = outputs[1]
        else:
            raise RuntimeError("Unknown outputs format")

        text_mask = text_mask.numpy()
        # TODO: configurable score
        # TODO: mask to polygon
        boxes, scores = utils.mask_to_box(text_mask, min_score=0.2)
        polygons = []
        for (x1, y1, x2, y2) in boxes:
            polygon = [(x1, y1),
                       (x1, y2),
                       (x2, y2),
                       (x2, y1)]
            polygons.append(polygon)

        return Sample(
            image=image,
            boxes=polygons,
            classes=np.zeros_like(scores).astype(int).tolist(),
            scores=scores.tolist(),
        )

    def compute_loss(self, outputs, targets):
        # Unpack GT and PR
        # text threshold, text probs, noise probs
        gt_tt, gt_tp, gt_np = targets.chunk(targets.shape[-3], dim=-3)
        pr_bg, pr_tt, pr_tp, pr_np = outputs.chunk(outputs.shape[-3], dim=-3)

        # Basic segmentation loss
        gt_bg = torch.ones_like(gt_tt)
        gt_bg = gt_bg & (~(gt_tt | gt_tp | gt_np))
        _targets = torch.cat([gt_bg, gt_tt, gt_tp, gt_np], dim=1) * 1.0
        loss = self.contour_loss(outputs, _targets)
        loss = F.mse_loss(outputs, _targets) + loss
        loss = F.binary_cross_entropy_with_logits(outputs, _targets) + loss
        loss = loss / 3

        # Loss for bg, text and text threshold
        gt_bg = torch.ones_like(gt_tt)
        gt_bg = gt_bg & (~gt_tt) & (~gt_tp)
        _outputs = torch.cat([pr_bg.detach(), pr_tt, pr_tp], dim=1)
        _targets = torch.cat([gt_bg, gt_tt, gt_tp], dim=1) * 1.0
        c_loss_t = F.cross_entropy(_outputs, _targets)

        # Loss for bg and noise
        gt_bg = torch.ones_like(gt_tt)
        gt_bg = gt_bg & (~gt_np)
        _outputs = torch.cat([pr_bg.detach(), pr_np], dim=1)
        _targets = torch.cat([gt_bg, gt_np], dim=1) * 1.0
        c_loss_n = F.cross_entropy(_outputs, _targets)

        # Loss for text and noise
        _gt_np = gt_np & (~gt_tt) & (~gt_tp)
        _pr_np = pr_np - (pr_tp + pr_tt) / 2
        _outputs = torch.cat([pr_tt, pr_tp, _pr_np], dim=1)
        _targets = torch.cat([gt_tt, gt_tp, _gt_np], dim=1) * 1.0
        c_loss_tn = F.cross_entropy(_outputs, _targets)

        loss = loss + c_loss_t + c_loss_n + c_loss_tn
        return loss
