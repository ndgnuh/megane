from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class DetrOutput:
    class_logits: Tensor
    boxes: Tensor
    loss: Optional[Tensor] = None

    def __post_init__():
        class_probs = torch.softmax(class_logits, dim=-1)
        self.classes, self.class_scores = class_probs.max(dim=-1)


class DetrLoss(nn.Module):
    def forward(self, pr_boxes, gt_boxes, pr_class_logits, gt_classes):
        pass


class DetrHead(nn.Module):
    def __init__(self, head_dims, num_classes: int):
        super().__init__()
        self.box_dims = 4
        self.classify_head = nn.Linear(head_dims, num_classes)
        self.localize_head = nn.Linear(head_dims, self.box_dims)
        self.loss = DetrLoss()

    def forward(self, features, batch):
        class_logits = self.classify_head(features)
        boxes = self.localize_head(features)
        if batch.get("boxes", None) is not None:
            loss = self.loss(boxes, batch["boxes"], class_logits, batch["classes"])
        else:
            loss = None

        return class_logits, boxes
