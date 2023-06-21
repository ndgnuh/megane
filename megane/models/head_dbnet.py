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


def encode_sample(
    sample: Sample,
    shrink_rate: float,
    num_classes: int,
    image_size: int,
):
    # Process inputs
    image = utils.prepare_input(
        sample.image,
        image_size,
        image_size,
    )

    # De-normalize boxes
    boxes = utils.denormalize_polygon(
        sample.boxes,
        image_size,
        image_size,
        batch=True,
    )

    # Calculate offset boxes
    areas = map(utils.polygon_area, boxes)
    lengths = map(utils.polygon_perimeter, boxes)
    dists = [(1 - shrink_rate**2) * A / L for A, L in zip(areas, lengths)]
    expand_boxes = starmap(utils.offset_polygon, zip(boxes, dists))
    dists = map(lambda x: -x, dists)
    shrink_boxes = starmap(utils.offset_polygon, zip(boxes, dists))

    # Helper function to filter boxes
    def filter_boxes(orig_boxes, class_idx):
        return [
            np.array(box).astype(int)
            for (box, class_idx_) in zip(orig_boxes, sample.classes)
            if class_idx == class_idx_
        ]

    # Draw target masks
    probas = []
    thresholds = []
    for class_idx in range(num_classes):
        # Filter boxes by class
        shrink_boxes_c = filter_boxes(shrink_boxes, class_idx)
        expand_boxes_c = filter_boxes(expand_boxes, class_idx)

        # Draw probability mask
        proba_map = utils.draw_mask(image_size, image_size, shrink_boxes_c)
        proba_map = np.clip(proba_map, 0, 1)
        probas.append(proba_map)

        # Draw threshold masks
        threshold_map = utils.draw_mask(image_size, image_size, expand_boxes_c)
        threshold_map = threshold_map - proba_map
        threshold_map = np.clip(threshold_map, 0, 1)
        thresholds.append(threshold_map)

    # Stack target maps
    probas = np.stack(probas, axis=0)
    thresholds = np.stack(thresholds, axis=0)

    return image, (probas, thresholds)


def SimpleFilter(channels):
    return nn.Conv2d(channels, channels, 3, padding=1, groups=channels)


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


class DBNetHead(DBNetFamily):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        image_size: int,
        expand_rate: float = 1.5,
        shrink_rate: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.expand_rate = expand_rate
        self.shrink_rate = shrink_rate
        self.thresholds = nn.Sequential(
            PredictionConv(hidden_size, num_classes),
        )
        self.probas = nn.Sequential(
            PredictionConv(hidden_size, num_classes),
        )

    def forward(self, features, targets=None):
        thresholds = self.thresholds(features)
        probas = self.probas(features)
        return (probas, thresholds)

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
        binmaps = self.db(probas, thresholds)
        if not ground_truth:
            probas = torch.sigmoid(probas)
            thresholds = torch.sigmoid(thresholds)

        probas = utils.stack_image_batch(probas)
        thresholds = utils.stack_image_batch(thresholds)
        binmaps = utils.stack_image_batch(binmaps)
        images = torch.cat([probas, thresholds, binmaps], dim=-1)

        logger.add_image(tag, images, step)

    def db(self, proba, thresh, k=50.0, logits=True):
        x = k * (proba - thresh)
        if logits:
            return torch.sigmoid(x)
        else:
            return x

    def hnm(self, batch_losses, masks, k=3):
        loss = 0
        for losses, positives in zip(batch_losses, masks):
            negatives = ~positives
            num_positives = torch.count_nonzero(positives)
            num_negatives = torch.count_nonzero(negatives)
            num_negatives = torch.minimum(num_positives * k, num_negatives)
            loss = (
                loss
                + losses[positives].sort(descending=True).values[:num_positives].mean()
                + losses[negatives].sort(descending=True).values[:num_negatives].mean()
            )
        return loss / losses.shape[0]

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets
        # training_mask = (gt_probas + gt_thresholds) > 0

        # Loss functions
        bce_logits = F.binary_cross_entropy_with_logits

        # Binary map loss
        pr_bin = self.db(pr_probas, pr_thresholds, logits=False)
        gt_bin = self.db(gt_probas, gt_thresholds)
        loss_bin = bce_logits(pr_bin, gt_bin * 1.0)

        # Proba map loss
        loss_proba = bce_logits(pr_probas, gt_probas * 1.0)
        loss_proba = F.l1_loss(torch.sigmoid(pr_probas), gt_probas * 1.0) + loss_proba

        # Threshold map loss
        loss_threshold = F.l1_loss(pr_thresholds, gt_thresholds * 1.0)

        loss = loss_bin + loss_proba + 10 * loss_threshold
        return loss


class HeadSegment(DBNetFamily):
    def __init__(
        self,
        hidden_size: int,
        image_size: int,
        num_classes: int,
        shrink_rate: float = 0.4,
        expand_rate: float = 1.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.expand_rate = expand_rate
        self.shrink_rate = shrink_rate
        self.outputs = PredictionConv(hidden_size, num_classes + 1)
        self.contour = losses.ContourLoss()

    def forward(self, features, tagets=None):
        outputs = self.outputs(features)
        return outputs

    def compute_loss(self, outputs, targets):
        loss_1 = F.cross_entropy(outputs, targets)
        targets = F.one_hot(targets, self.num_classes + 1).transpose(-1, 1)
        loss_2 = self.contour(outputs, targets * 1.0)

        loss = (loss_1 + loss_2) / 2
        return loss

    def encode_sample(self, sample):
        image_size = self.image_size
        shrink_rate = self.shrink_rate
        num_classes = self.num_classes

        # process inputs
        image = utils.prepare_input(
            sample.image,
            image_size,
            image_size,
        )

        # de-normalize boxes
        boxes = [
            [(x * image_size, y * image_size) for (x, y) in polygon]
            for polygon in sample.boxes
        ]

        # calculate offset boxes
        areas = [utils.polygon_area(poly) for poly in boxes]
        lengths = [utils.polygon_perimeter(poly) for poly in boxes]
        dists = [(1 - shrink_rate**2) * a / l for (a, l) in zip(areas, lengths)]
        shrink_boxes = [
            utils.offset_polygon(box, -distance)
            for (box, distance) in zip(boxes, dists)
        ]

        # Helper function to filter boxes
        def filter_boxes(orig_boxes, class_idx):
            return [
                np.array(box).astype(int)
                for (box, class_idx_) in zip(orig_boxes, sample.classes)
                if class_idx == class_idx_
            ]

        # Draw target mask
        target = np.zeros([image_size, image_size], dtype="float32")
        for class_idx in range(num_classes - 1, 0, -1):
            for box, class_idx_ in zip(shrink_boxes, sample.classes):
                if class_idx_ != class_idx_:
                    continue
                cv2.fillConvexPoly(
                    target, np.array(box).astype(int), float(class_idx + 1)
                )
        target = target.astype(int)

        return image, target

    @torch.no_grad()
    def visualize_outputs(self, outputs, logger, tag, step, ground_truth: bool = False):
        outputs = outputs.cpu()
        if ground_truth:
            outputs = outputs.unsqueeze(1)
        else:
            outputs = torch.clip(torch.sigmoid(50 * outputs), 0, 1)

        outputs = utils.stack_image_batch(outputs)

        logger.add_image(tag, outputs, step)


class DBNetHeadForDetection(DBNetFamily):
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
