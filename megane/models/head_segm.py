from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .. import utils
from ..data import Sample
from .api import ModelAPI


def encode_sample(sample: Sample, num_classes, image_size, output_size):
    image = utils.prepare_input(sample.image, image_size, image_size)
    classes = sample.classes
    boxes = [(np.array(box) * output_size).astype(int) for box in sample.boxes]
    masks = []
    for c in range(num_classes):
        boxes_c = [box for box, cls in zip(boxes, classes) if cls == c]
        mask = utils.draw_mask_v2(output_size, output_size, boxes_c)
        masks.append(mask)
    masks = np.stack(masks, axis=0)
    return image, masks


def decode_sample(inputs, outputs, ground_truth=False) -> Sample:
    pass


class SegmentHead(ModelAPI):
    def __init__(
        self, num_classes: int, hidden_size: int, image_size: int, output_size: int
    ):
        super().__init__()
        self.image_size = image_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.bg_output = nn.Sequential(
            nn.InstanceNorm2d(hidden_size),
            nn.Conv2d(hidden_size, 1, 1, bias=False),
            nn.Upsample(size=output_size),
        )
        self.output = nn.Sequential(
            nn.InstanceNorm2d(hidden_size),
            nn.Conv2d(hidden_size, num_classes, 1, bias=False),
            nn.Upsample(size=output_size),
        )

    def forward(self, features, targets=None):
        bg = self.bg_output(features)
        outputs = self.output(features)
        outputs = torch.cat([bg, outputs], dim=1)
        return outputs

    def encode_sample(self, sample: Sample):
        return encode_sample(
            sample, self.num_classes, self.image_size, self.output_size
        )

    @torch.no_grad()
    def decode_sample(self, inputs, outputs, ground_truth=False):
        outputs = outputs.detach().cpu()
        all_boxes = []
        all_scores = []
        all_classes = []
        for i, output in enumerate(outputs):
            if ground_truth:
                mask = output.numpy()
            else:
                mask = torch.clip(torch.tanh(output), 0, 1).numpy()
            boxes, scores = utils.mask_to_polygon(mask)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend([i] * len(scores))

        image = TF.to_pil_image(inputs)
        return Sample(
            image=image, boxes=all_boxes, scores=all_scores, classes=all_classes
        )

    @torch.no_grad()
    def visualize_outputs(self, outputs):
        try:
            outputs = torch.clip(torch.tanh(outputs), 0, 1)
            outputs = torch.cat([o for o in outputs], dim=-1)
            outputs = torch.cat([o for o in outputs], dim=-2)
            return outputs.unsqueeze(0)
        except Exception:
            return torch.zeros([1, 10, 10])

    def compute_loss(self, outputs, targets):
        # Generate background mask
        bg_targets = torch.ones_like(targets[:, 0], dtype=torch.bool).unsqueeze(1)
        for tg in targets.chunk(targets.shape[1], dim=1):
            bg_targets = bg_targets & (~tg.type(torch.bool))

        # Add background to GT
        targets = torch.cat([bg_targets, targets], dim=1)
        return F.cross_entropy(outputs, targets)
