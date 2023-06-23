# Reference: https://arxiv.org/abs/1911.08947
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import utils
from megane.data import Sample
from megane.models.api import ModelAPI


def generate_background(x, dim=-3, logit=False):
    if logit:
        x = torch.sigmoid(x)
    bg = 1 - x.sum(dim=dim, keepdim=True)
    bg = torch.clamp(bg, 1e-6, 1 - 1e-6)
    if logit:
        bg = torch.log(bg / (1 - bg))
    return bg


def with_background(x, dim=-3, logit=False):
    bg = generate_background(x, dim=dim, logit=logit)
    return torch.cat([bg, x], dim=dim)


class PredictionConv(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        aux_size = hidden_size // 4
        self.conv_1 = nn.Sequential(
            nn.Conv2d(hidden_size, aux_size, 3, padding=1),
            nn.InstanceNorm2d(aux_size),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(
                aux_size,
                aux_size,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(aux_size),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(aux_size, 3, kernel_size=4,
                               stride=2, padding=1),
            nn.Conv2d(3, 3, 3, padding=1, groups=3),
            nn.Conv2d(3, num_classes, 1),
        )

        self.proj = nn.Conv2d(hidden_size, aux_size, 1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        res = self.proj(x)
        x = self.conv_1(x) + res
        res = self.up(res)
        x = self.conv_2(x) + res
        x = self.conv_3(x)
        return x


class DBNet(ModelAPI):
    def __init__(
        self,
        image_size: int,
        hidden_size: int,
        num_classes: int,
        shrink_rate: float = 0.4,
        expand_rate: float = 1.5,
        resize_mode: str = "resize",
    ):
        super().__init__()
        utils.save_args()

        # 0 = background
        self.probas = PredictionConv(hidden_size, num_classes)
        # 0 = threshold
        self.thresholds = PredictionConv(hidden_size, num_classes)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            thresholds = None
        else:
            thresholds = self.thresholds(features)
        return (probas, thresholds)

    def db(self, P, T, k=50, logits=True):
        x = k * (P - T)
        if logits:
            return torch.sigmoid(x)
        else:
            return x

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets

        def bbce(pr, gt):
            pos = gt > 0.2
            neg = ~pos
            if pos.count_nonzero() == 0 or neg.count_nonzero() == 0:
                loss = F.cross_entropy(pr, gt)
            else:
                p_losses = F.cross_entropy(pr[pos], gt[pos])
                n_losses = F.cross_entropy(pr[neg], gt[neg])
                loss = (p_losses + n_losses) / 2
            return loss

        loss = 0
        count = 0
        l1 = F.l1_loss
        for i in range(self.num_classes):
            # Prepare
            pr_proba = pr_probas[:, i]
            pr_threshold = pr_thresholds[:, i]
            gt_proba = gt_probas[:, i]
            gt_threshold = gt_thresholds[:, i]

            # Full classification loss
            pr = with_background(
                torch.stack([pr_proba, pr_threshold], dim=1),
                logit=True,
            )
            gt = with_background(
                torch.stack([gt_proba, gt_threshold], dim=1),
                logit=False,
            )
            loss += F.cross_entropy(pr, gt)

            # Binary map loss
            # pr_bin = self.db(pr_proba, pr_threshold, logits=False)
            # gt_bin = self.db(gt_proba, gt_threshold, logits=True)
            # loss += F.binary_cross_entropy_with_logits(pr_bin, gt_bin)

            # Proba map loss
            pr = with_background(pr_proba.unsqueeze(1))
            gt = with_background(gt_proba.unsqueeze(1))
            loss += F.binary_cross_entropy_with_logits(pr, gt)

            # Threshold map loss
            pr = with_background(pr_threshold.unsqueeze(1))
            gt = with_background(gt_threshold.unsqueeze(1))
            loss += F.binary_cross_entropy_with_logits(pr, gt)

            count = count + 4

        loss = loss / count
        return loss

    def encode_sample(self, sample: Sample):
        sz = self.image_size
        num_classes = self.num_classes
        r = self.shrink_rate

        # Process inputs
        image = utils.prepare_input(
            sample.image, sz, sz, resize_mode=self.resize_mode)

        # Process targets
        # Expand polygons
        boxes = utils.denormalize_polygon(sample.boxes, sz, sz, batch=True)
        areas = map(utils.polygon_area, boxes)
        lengths = map(utils.polygon_perimeter, boxes)
        dists = [(1 - r**2) * A / (L + 1e-6) for (A, L) in zip(areas, lengths)]
        expand_boxes = [utils.offset_polygon(
            b, d) for b, d in zip(boxes, dists)]
        shrink_boxes = [utils.offset_polygon(
            b, -d) for b, d in zip(boxes, dists)]

        # Numpy makes it easy to filter
        shrink_boxes = np.array(shrink_boxes, dtype="object")
        expand_boxes = np.array(expand_boxes, dtype="object")
        classes = np.array(sample.classes, dtype="object")

        # Draw target masks
        probas = []
        thresholds = []
        for class_idx in range(num_classes):
            # Filter boxes by classes
            class_mask = class_idx == classes
            expand_boxes_c = expand_boxes[class_mask]
            shrink_boxes_c = shrink_boxes[class_mask]

            # Draw masks
            proba = utils.draw_mask(sz, sz, shrink_boxes_c)
            thresh = utils.draw_mask(sz, sz, expand_boxes_c) - proba
            thresh = np.clip(thresh, 0, 1)

            # Add result
            probas.append(proba)
            thresholds.append(thresh)

        # Stack
        probas = np.stack(probas, axis=0)
        thresholds = np.stack(thresholds, axis=0)

        return image, (probas, thresholds)

    def post_process(self, probas):
        return torch.clamp(torch.tanh(probas * 50), 0, 1)

    @torch.no_grad()
    def visualize_outputs(
        self,
        outputs,
        logger,
        tag,
        step,
        ground_truth: bool = False,
    ):
        probas, thresholds = outputs
        images = torch.cat([probas, thresholds], dim=-3).cpu()
        if not ground_truth:
            images = self.post_process(images * 50)

        images = utils.stack_image_batch(images)
        logger.add_image(tag, images, step)

    @torch.no_grad()
    def decode_sample(self, inputs, outputs, ground_truth: bool = False):
        image = TF.to_pil_image(inputs.detach().cpu())

        # post process
        outputs = outputs[0].detach().cpu()
        if not ground_truth:
            outputs = self.post_process(outputs)

        #  adapt
        masks = outputs.numpy()

        # decode
        final_classes = []
        final_scores = []
        final_polygons = []
        r = self.expand_rate
        for class_idx in range(self.num_classes):
            mask = masks[class_idx]
            polygons, scores = utils.mask_to_polygons(mask)
            if r > 0:
                areas = utils.polygon_area(polygons, batch=True)
                lengths = utils.polygon_perimeter(polygons, batch=True)
                dists = [A * r / L for (A, L) in zip(areas, lengths)]
                polygons = [
                    utils.offset_polygon(p, d) for (p, d) in zip(polygons, dists)
                ]

            final_polygons.extend(polygons)
            final_classes.extend([class_idx] * len(polygons))
            final_scores.extend(scores)

        # output sample
        sample = Sample(
            image=image,
            boxes=final_polygons,
            scores=final_scores,
            classes=final_classes,
        )
        return sample
