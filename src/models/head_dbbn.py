from typing import *

import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torch import nn


from .. import utils
from ..data import Sample
from ..configs import ModelConfig
from .losses import ContourLoss
from .api import ModelAPI


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
    images = torch.sigmoid((outputs * 1.0 - 0.5) * 1e6)

    # Stack batches to height
    images = torch.cat([image for image in images], dim=-2)

    # Stack channels to width
    images = torch.cat([image for image in images], dim=-1)

    # To [C, H, W]
    images = images.unsqueeze(0)
    return images


class DBBNHead(ModelAPI):
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
        self.contour_loss = ContourLoss(
            kernel_size=config.head.contour_loss_kernel)

    def visualize_outputs(self, outputs, ground_truth: bool = False):
        return visualize_outputs(outputs)

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
        image = utils.prepare_input(
            sample.image, self.image_size, self.image_size)
        image = torch.FloatTensor(image)

        # Compute shrink/expand distance
        # boxes = [np.array(b, dtype='float32') for b in sample.boxes]
        boxes = [
            [(x * output_size, y * output_size) for (x, y) in polygon]
            for polygon in sample.boxes
        ]
        shrink = []
        expand = []
        r = 0.4
        for box in boxes:
            A = utils.polygon_area(box)
            L = utils.polygon_perimeter(box)
            D = (1 - r**2) * A / L
            s_box = utils.offset_polygon(box, -D)
            e_box = utils.offset_polygon(box, D)
            shrink.append(s_box)
            expand.append(e_box)

        # Box rounding
        boxes = [np.array(box, dtype=int) for box in boxes]
        shrink = [np.array(box, dtype=int) for box in shrink]
        expand = [np.array(box, dtype=int) for box in expand]

        # Mask draw helper
        classes = sample.classes

        def dmask(boxes, c_idx):
            boxes_by_class = [boxes[i]
                              for (i, c) in enumerate(classes) if c == c_idx]
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

    def decode_sample(
        self,
        inputs,
        outputs,
        sample: Optional[Sample] = None,
        ground_truth: bool = False,
    ) -> Sample:
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
            ground_truth:
                Specify if this is decoded from the ground truth

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

        # TODO: configurable score
        # TODO: mask to polygon

        if ground_truth:
            text_mask = text_mask.numpy()
        else:
            text_mask = torch.clip(torch.tanh(text_mask), 0, 1).numpy()
        mask_size = self.image_size // self.final_div_factor
        boxes, scores = utils.mask_to_polygon(text_mask)
        polygons = []
        final_scores = []
        for polygon, score in zip(boxes, scores):
            if score < 0.5:
                continue
            area = utils.polygon_area(polygon)
            length = utils.polygon_perimeter(polygon)
            if length == 0 or area == 0:
                continue
            d = area * 1.5 / length
            polygon = utils.offset_polygon(polygon, d)
            polygon = np.clip(polygon, 0, mask_size)
            polygon = [(x / mask_size, y / mask_size) for (x, y) in polygon]
            polygons.append(polygon)
            final_scores.append(score)

        return Sample(
            image=image,
            boxes=polygons,
            classes=np.zeros_like(final_scores).astype(int).tolist(),
            scores=final_scores,
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
