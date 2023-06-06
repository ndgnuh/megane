from typing import *

import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torch import nn


from .. import utils
from ..data import Sample
from ..configs import ModelConfig


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

        # Background and threshold
        self.heads = nn.ModuleList([nn.Conv2d(self.hidden_size, 1, 1) for _ in range(4)])
        self.inferring = False

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
        boxes, scores = utils.mask_to_box(text_mask, min_score=0.5)
        boxes = utils.expand(boxes, 1.5)
        normalizer = np.array([w, h, w, h]).reshape(1, 4)
        boxes = (boxes * normalizer).round().astype(int)
        from tqdm import tqdm

        return Sample(
            image=image,
            boxes=boxes,
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
        loss = F.l1_loss(outputs, _targets)
        loss = F.mse_loss(outputs, _targets) + loss
        loss = loss / 2

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
