from dataclasses import dataclass
from typing import List, Tuple

import cv2
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import registry
from megane.data import Sample
from megane.models import losses as L
from megane.models.head_rrm import ResidualRefinementModule


@torch.jit.script
def polygon2xyxy_single(polygon: List[Tuple[float, float]]):
    x = [pt[0] for pt in polygon]
    y = [pt[1] for pt in polygon]
    return [int(min(x)), int(min(y)), int(max(x)), int(max(y))]


@torch.jit.script
def polygon2xyxy(polygons: List[List[Tuple[float, float]]]):
    return [polygon2xyxy_single(polygon) for polygon in polygons]


@torch.jit.script
def encode_clsm_straight(
    W: int,
    H: int,
    C: int,
    boxes: List[Tuple[int, int, int, int]],
    classes: List[int],
):
    targets = torch.zeros((C, H, W), dtype=torch.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = classes[i]
        targets[c, y1:y2, x1:x2] = 1

    return targets


@torch.jit.script
def xyxy2polygon(box: Tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


@torch.jit.script
def scale_polygons(polygons: List[List[Tuple[float, float]]], W: int, H: int):
    return [[(x * W, y * H) for (x, y) in box] for box in polygons]


@registry.target_encoders.register(name="clsm")
@dataclass
class CLSMEncoder:
    num_classes: int
    straight: bool

    def __call__(self, sample: Sample):
        # Unpack
        W, H = sample.image.size
        C = self.num_classes
        boxes = sample.boxes
        boxes = scale_polygons(boxes, W, H)
        boxes = polygon2xyxy(boxes)
        classes = sample.classes

        # Encode
        images = TF.to_tensor(sample.image)
        targets = encode_clsm_straight(W, H, C, boxes, classes)
        return images, targets


@registry.target_decoders.register(name="clsm")
@dataclass
class CLSMDecoder:
    straight: bool

    def __call__(self, inputs, outputs, ground_truth=False):
        # Post process
        if not ground_truth:
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs = torch.sigmoid(outputs)

        # Decode image
        image = TF.to_pil_image(inputs)
        W, H = image.size

        outputs = outputs.detach().cpu().numpy()
        boxes = []
        classes = []
        for i in range(outputs.shape[-3]):
            mask = (outputs[i] * 255).astype("uint8")
            stats = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, s in stats[2]:
                box = (x / W, y / H, (x + w) / W, (y + h) / H)
                box = xyxy2polygon(box)
                boxes.append(box)
                classes.append(i)
        return Sample(image, classes=classes, boxes=boxes)


@registry.heads.register(name="clsm")
class ClassifierSegmenter(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, num_classes, 4, stride=4),
        )
        self.conv_reflect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, num_classes, 4, stride=4),
        )

    def forward(self, ft, *args, **kwargs):
        x = ft
        x = self.conv_1(x)
        if self.infer:
            return torch.sigmoid(x)
        else:
            reflect = self.conv_reflect(ft)
            return x, reflect

    def compute_loss(self, outputs, targets):
        probas, reflects = outputs

        # Informations
        loss = 0
        pos = (probas > 0.5) == (targets > 0)
        neg = ~pos
        gt_corrects = pos * 1.0
        bce = F.binary_cross_entropy_with_logits
        bsize = probas.shape[0]

        # Reflection loss
        losses = []
        for i in range(bsize):
            loss = 0

            r_loss = 0
            r_loss += bce(reflects[pos], gt_corrects[pos])
            r_loss += bce(reflects[neg], gt_corrects[neg])
            r_loss = r_loss / 2

            # Mask loss
            m_loss = 0
            m_loss += bce(probas[pos], targets[pos])
            m_loss += bce(probas[neg], targets[neg])
            m_loss = m_loss / 2

            # Total loss
            loss = (m_loss + r_loss) / 2
            losses.append(loss)

        k = max(bsize // 2, 1)
        losses = torch.topk(torch.stack(losses), k=k).values
        loss = losses.mean()
        return loss

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth=False):
        if not ground_truth:
            outputs = torch.sigmoid(outputs[0])
        else:
            ic(outputs.shape)

        if outputs.ndim == 4:
            outputs = torch.cat(list(outputs), dim=-2)
        outputs = torch.cat(list(outputs), dim=-1)
        outputs = outputs.unsqueeze(0)
        logger.add_image(tag, outputs, step)
