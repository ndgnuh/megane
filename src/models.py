from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import torch
import numpy as np
from shapely.geometry import Polygon
from torch import nn
from torchvision.models import mobilenet_v3_small

from .structures.configs import ModelConfig


def get_matching_index(pr_boxes: np.ndarray, gt_boxes: np.ndarray):
    pr_polygons = [Polygon(box) for box in pr_boxes]
    gt_polygons = [Polygon(box) for box in gt_boxes]
    n, m = len(pr_polygons), len(gt_polygons)

    # Calculate scoring matrix
    iou_matrix = np.zeros([n, m])
    for i in range(n):
        pr = pr_polygons[i]
        if not pr.is_valid:
            continue

        for j in range(m):
            gt = gt_polygons[j]
            inter = pr.intersection(gt).area
            uni = pr.union(gt).area
            iou_matrix[i, j] = inter / uni

    # Get the maximal matching based on score matrix
    mask = np.zeros(n, bool)
    gt_order = np.zeros(m, int)
    count = 0
    while count < m:
        idx = np.argmax(iou_matrix)
        i, j = np.unravel_index(idx, (n, m))

        mask[i] = True
        gt_order[count] = j
        iou_matrix[i, :] = -np.inf
        iou_matrix[:, j] = -np.inf

        count += 1
    return mask, gt_order


class HungarianMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def to_np(self, x):
        return x.detach().cpu().numpy()

    def forward(self, pr_boxes, gt_boxes, pr_classes_logits, gt_classes):
        device = pr_boxes.device
        l_loss = 0
        c_loss = 0

        count = 0
        for pr_boxes_i, gt_boxes_i, pr_classes_logits_i, gt_classes_i in zip(
            pr_boxes, gt_boxes, pr_classes_logits, gt_classes
        ):
            count = count + 1
            pr_boxes_np = self.to_np(pr_boxes_i)
            gt_boxes_np = self.to_np(gt_boxes_i)
            mask, gt_order = get_matching_index(pr_boxes_np, gt_boxes_np)
            mask = torch.BoolTensor(mask).to(device)
            gt_order = torch.LongTensor(gt_order).to(device)

            # Localization loss
            neg = pr_boxes_i[~mask]
            l_loss += self.ce(pr_boxes_i[mask], gt_boxes_i[gt_order])
            l_loss += self.ce(neg, torch.zeros_like(neg))

            # Classification loss
            neg = pr_classes_logits_i[mask]
            # ic(type(gt_classes), np.array(gt_classes).shape)
            c_loss += self.ce(pr_classes_logits_i[mask], gt_classes_i[gt_order])
            c_loss += self.ce(neg, torch.zeros_like(neg))

        loss = l_loss + c_loss
        loss = loss / count / 2
        return loss


class MeganeDetector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.feature = mobilenet_v3_small(num_classes=1).features
        self.num_classes = config.num_classes  # background
        self.num_box_dims = config.num_box_dims

        self.classify_head = nn.Conv2d(
            self.num_feature_channels, self.num_classes, 1, bias=False
        )
        self.localize_head = nn.Conv2d(
            self.num_feature_channels, self.num_box_dims, 1, bias=False
        )
        self.loss = HungarianMatchingLoss()

    @cached_property
    def num_feature_channels(self):
        with torch.no_grad():
            image = torch.rand(1, 3, 256, 256)
            feature_channels = self.feature(image).shape[1]
        return feature_channels

    def forward(self, batch):
        images = batch["images"]
        # extract features
        feature = self.feature(images)
        # predictions
        class_logits = self.classify_head(feature)
        boxes = self.localize_head(feature)

        # b c h w -> b c (h w) -> b (h w) c
        class_logits = class_logits.flatten(-2).transpose(-1, -2)
        boxes = boxes.flatten(-2).transpose(-1, -2).reshape(images.shape[0], -1, 4, 2)

        # Calculate loss
        if "classes" in batch:
            loss = self.loss(boxes, batch["boxes"], class_logits, batch["classes"])
        else:
            loss = None

        return DetrDetectionOutput(class_logits=class_logits, boxes=boxes, loss=loss)


@dataclass
class DetrDetectionOutput:
    class_logits: torch.Tensor
    boxes: torch.Tensor
    loss: Optional[torch.Tensor] = None
