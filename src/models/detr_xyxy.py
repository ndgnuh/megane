from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import ops


@dataclass
class DetrOutput:
    class_logits: Tensor
    boxes: Tensor
    loss: Optional[Tensor] = None
    classes: Tensor = None
    class_scores: Tensor = None

    def __post_init__(self):
        if self.classes is None:
            class_probs = torch.softmax(self.class_logits, dim=-1)
            self.classes, self.class_scores = class_probs.max(dim=-1)

    def __getitem__(self, i):
        return DetrOutput(
            class_logits=self.class_logits[i],
            classes=self.classes[i],
            class_scores=self.class_scores[i],
            boxes=self.boxes[i],
            loss=None,
        )


class DetrLoss(nn.Module):
    def get_matching(self, pr_boxes, gt_boxes):
        """
        Return maximal IoU matching based on score matrix, which includes PR mask and GT permutation
        """
        ious = ops.box_iou(pr_boxes, gt_boxes)
        n, m = ious.shape

        pr_mask = torch.zeros(n, dtype=torch.bool)
        gt_order = torch.arange(m)
        count = 0
        while count < m:
            # Find max
            idx = torch.argmax(ious)
            i, j = idx // m, idx % m

            # assign
            pr_mask[i] = True
            gt_order[count] = j

            # Mask score
            ious[i, :] = -torch.inf
            ious[:, j] = -torch.inf

            # next
            count = count + 1
        return pr_mask, gt_order

    def forward(self, pr_boxes, gt_boxes, pr_class_logits, gt_classes):
        l_loss = 0
        c_loss = 0
        count = 0

        iterator = zip(pr_boxes, gt_boxes, pr_class_logits, gt_classes)
        for item in iterator:
            pr_boxes_i, gt_boxes_i, pr_class_logits_i, gt_classes_i = item

            pr_mask, gt_order = self.get_matching(pr_boxes_i, gt_boxes_i)

            # Localization loss
            pr_pos = pr_boxes_i[pr_mask]
            pr_neg = pr_boxes_i[~pr_mask]
            gt_pos = gt_boxes_i[gt_order]
            gt_neg = torch.zeros_like(pr_neg)
            l_loss = l_loss + ops.generalized_box_iou_loss(pr_pos, gt_pos, reduction="mean")
            l_loss = l_loss + ops.generalized_box_iou_loss(pr_neg, gt_neg, reduction="mean")

            # classification loss
            pr_pos = pr_class_logits_i[pr_mask].transpose(0, -1).unsqueeze(0)
            pr_neg = pr_class_logits_i[~pr_mask].transpose(0, -1).unsqueeze(0)
            gt_pos = gt_classes_i[gt_order].unsqueeze(0)
            gt_neg = torch.zeros(pr_neg.shape[:-1], device=pr_neg.device, dtype=torch.long)
            ic(pr_pos.shape, gt_pos.shape, pr_pos.dtype, gt_pos.dtype)
            # ic(pr_neg.shape, gt_neg.shape, gt_neg.dtype)
            # c_loss = c_loss + F.cross_entropy(pr_pos, gt_pos)
            # c_loss += F.cross_entropy(pr_neg, gt_neg)

            # Add counting
            count = count + 1

        loss = (c_loss + l_loss) / count / 2
        return loss


class DetrHead(nn.Module):
    def __init__(self, head_dims, num_classes: int):
        super().__init__()
        self.box_dims = 4
        self.classify_head = nn.Linear(head_dims, num_classes)
        self.localize_head = nn.Sequential(
            nn.Linear(head_dims, self.box_dims),
            nn.Sigmoid(),
        )
        self.loss = DetrLoss()

    def forward(self, features, batch):
        class_logits = self.classify_head(features)
        boxes = self.localize_head(features)
        if batch.get("boxes", None) is not None:
            loss = self.loss(boxes, batch["boxes"], class_logits, batch["classes"])
        else:
            loss = None

        return DetrOutput(class_logits=class_logits, boxes=boxes, loss=loss)
