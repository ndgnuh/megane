from itertools import starmap

# Reference: https://arxiv.org/abs/1911.08947
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import utils
from megane.data import Sample
from megane.models import api, losses


def PredictionConv(hidden_size, num_classes: int = 1):
    aux_size = hidden_size // 4
    return nn.Sequential(
        nn.Conv2d(hidden_size, aux_size, 3, padding=1),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        nn.ConvTranspose2d(aux_size, aux_size, kernel_size=2, stride=2),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        nn.ConvTranspose2d(aux_size, num_classes, kernel_size=2, stride=2),
    )


class DBNetFamily(api.ModelAPI):
    """Family of prediction heads that uses shrink/expand polygons"""

    def encode_sample(self, sample: Sample):
        return encode_sample(
            sample,
            shrink_rate=self.shrink_rate,
            image_size=self.image_size,
            num_classes=self.num_classes,
        )

    @torch.no_grad()
    def decode_sample(
        self,
        inputs,
        outputs,
        ground_truth: bool = False,
    ) -> Sample:
        """Return a sample from raw model outputs
        Args:
            inputs:
                Model encoded input image of shape [3, H, W]
            outputs:
                Tuple of proba map and threshold map
            sample:
                The input sample, this is only used to get the image.
            ground_truth:
                Specify if this is decoded from the ground truth

        Returns:
            The decoded sample.
        """
        image = TF.to_pil_image(inputs)
        probas = outputs[0].detach().cpu()

        # Mask to polygons
        if not ground_truth:
            probas = torch.sigmoid(probas)

        # For each classes
        final_classes = []
        final_polygons = []
        final_scores = []
        for class_idx, mask in enumerate(probas):
            mask_size = self.image_size
            boxes, scores = utils.mask_to_rrect(mask.numpy())
            for polygon, score in zip(boxes, scores):
                # Filter bounding boxes
                if score < 0.5:
                    continue
                area = utils.polygon_area(polygon)
                length = utils.polygon_perimeter(polygon)
                if length == 0 or area == 0:
                    continue

                # Expand detected polygon
                if self.expand_rate > 0:
                    d = area * self.expand_rate / length
                    polygon = utils.offset_polygon(polygon, d)
                    polygon = np.clip(polygon, 0, mask_size)
                polygon = [(x / mask_size, y / mask_size) for (x, y) in polygon]

                # Append result
                final_polygons.append(polygon)
                final_scores.append(score)
                final_classes.append(class_idx)

        return Sample(
            image=image,
            boxes=final_polygons,
            classes=final_classes,
            scores=final_scores,
        )


class DBNet(DBNetFamily):
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
        self.probas = PredictionConv(hidden_size, num_classes + 1)
        # 0 = threshold
        self.thresholds = PredictionConv(hidden_size, num_classes)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            thresholds = None
        else:
            thresholds = self.thresholds(features)
        return (probas, thresholds)

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets

        # Loss function
        if self.num_classes == 1:
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            loss_fn = F.multilabel_soft_margin_loss

        # Prepare
        pr = torch.cat([pr_probas, pr_thresholds], dim=1)
        gt = torch.cat([gt_probas, gt_thresholds], dim=1)
        if self.num_classes > 1:
            pr = pr.transpose(1, -1).reshape(-1, pr.size(1))
            gt = gt.transpose(1, -1).reshape(-1, gt.size(1))
        loss = loss_fn(pr, gt)

        # Additional loss between the threshold and the proba
        count = 0
        extra_loss = 0
        for i in range(self.num_classes):
            pr_proba = pr_probas[:, i + 1]
            pr_threshold = pr_thresholds[:, i]
            gt_proba = gt_probas[:, i + 1]
            gt_threshold = gt_thresholds[:, i]
            pr_bg = torch.clamp(1 - pr_threshold - pr_proba, 0, 1)
            gt_bg = torch.clamp(1 - gt_threshold - gt_proba, 0, 1)
            pr = torch.stack([pr_bg, pr_threshold, pr_proba], dim=1)
            gt = torch.stack([gt_bg, gt_threshold, gt_proba], dim=1)
            extra_loss = extra_loss + F.cross_entropy(pr, gt)

            # DB Loss
            k = 50
            pr = k * (gt_proba - gt_threshold)
            gt = torch.sigmoid(k * (gt_proba - gt_threshold))
            extra_loss = extra_loss + F.binary_cross_entropy_with_logits(pr, gt)
            count = count + 1

        loss = loss + extra_loss / count / 2
        return loss

    def encode_sample(self, sample: Sample):
        sz = self.image_size
        num_classes = self.num_classes
        r = self.shrink_rate

        # Process inputs
        image = utils.prepare_input(sample.image, sz, sz, resize_mode=self.resize_mode)

        # Process targets
        # Expand polygons
        boxes = utils.denormalize_polygon(sample.boxes, sz, sz, batch=True)
        areas = map(utils.polygon_area, boxes)
        lengths = map(utils.polygon_perimeter, boxes)
        dists = [(1 - r**2) * A / (L + 1e-6) for (A, L) in zip(areas, lengths)]
        expand_boxes = [utils.offset_polygon(b, d) for b, d in zip(boxes, dists)]
        shrink_boxes = [utils.offset_polygon(b, -d) for b, d in zip(boxes, dists)]

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

        # Make backgrounds
        backgrounds = 1 - np.clip(probas.sum(axis=0), 0, 1)
        probas = np.concatenate([backgrounds[None, ...], probas], axis=0)

        return image, (probas, thresholds)

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
            images = torch.sigmoid(images)

        images = utils.stack_image_batch(images)
        logger.add_image(tag, images, step)

    @torch.no_grad()
    def decode_sample(self, inputs, outputs, ground_truth: bool = False):
        image = TF.to_pil_image(inputs.detach().cpu())

        # post process
        outputs = outputs[0].detach().cpu()
        if not ground_truth:
            outputs = torch.softmax(outputs, dim=-3)

        masks, classes = outputs.max(dim=-3)

        #  adapt
        masks = masks.numpy()
        classes = classes.numpy()

        # decode
        final_classes = []
        final_scores = []
        final_polygons = []
        r = self.expand_rate
        for class_idx in range(self.num_classes):
            class_mask = classes == (class_idx + 1)
            mask = masks * class_mask
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
