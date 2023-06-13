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
    boxes = [
        [(x * image_size, y * image_size) for (x, y) in polygon]
        for polygon in sample.boxes
    ]

    # Calculate offset boxes
    areas = [utils.polygon_area(poly) for poly in boxes]
    lengths = [utils.polygon_perimeter(poly) for poly in boxes]
    dists = [(1 - shrink_rate**2) * A / L for (A, L) in zip(areas, lengths)]
    expand_boxes = [
        utils.offset_polygon(
            box,
            distance,
        )
        for (box, distance) in zip(boxes, dists)
    ]
    shrink_boxes = [
        utils.offset_polygon(
            box,
            -distance,
        )
        for (box, distance) in zip(boxes, dists)
    ]

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
        proba_map = np.zeros((image_size, image_size), dtype="float32")
        for box in shrink_boxes_c:
            cv2.fillConvexPoly(proba_map, box, 1)
        probas.append(proba_map)

        # Draw threshold masks
        threshold_map = np.zeros((image_size, image_size), dtype="float32")
        for inner_box, outer_box in zip(shrink_boxes_c, expand_boxes_c):
            # Draw to a canvas first
            # and then fill the inner box with background
            canvas = np.zeros_like(threshold_map)
            canvas = cv2.fillConvexPoly(canvas, outer_box, 1)
            canvas = cv2.fillConvexPoly(canvas, inner_box, 0)
            # yank the canvas to the threshold map
            threshold_map = threshold_map + canvas
        # Normalize threshold map to 0..1
        threshold_map = np.clip(threshold_map, 0, 1)
        thresholds.append(threshold_map)

    # Stack target maps
    probas = np.stack(probas, axis=0)
    thresholds = np.stack(thresholds, axis=0)

    return image, (probas, thresholds)


def PredictionConv(hidden_size, num_classes: int = 1):
    aux_size_1 = hidden_size // 4
    aux_size_2 = hidden_size // 8
    return nn.Sequential(
        nn.Conv2d(hidden_size, aux_size_1, 3, bias=False, padding=1),
        nn.InstanceNorm2d(aux_size_1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            aux_size_1,
            aux_size_2,
            kernel_size=2,
            stride=2,
            bias=False,
        ),
        nn.InstanceNorm2d(aux_size_2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            aux_size_2,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            bias=False,
        ),
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

    def decode_sample(self, inputs, outputs, ground_truth=False):
        image = TF.to_pil_image(inputs)
        return Sample(image=image)


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
        self.thresholds = self.make_head()
        self.probas = self.make_head()
        self.dice = losses.DiceLoss()

    def forward(self, features, targets=None):
        thresholds = torch.cat([conv(features) for conv in self.thresholds], dim=1)
        probas = torch.cat([conv(features) for conv in self.probas], dim=1)
        return (probas, thresholds)

    def make_head(self):
        return nn.ModuleList(
            [PredictionConv(self.hidden_size) for _ in range(self.num_classes)]
        )

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth: bool = False):
        if ground_truth:
            outputs.unsqueeze(1)
        else:
            outputs = torch.clip(torch.sigmoid(50 * outputs), 0, 1)
        outputs = utils.stack_image_batch(outputs)

        logger.add_image(tag, outputs, step)

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets

        # Background loss
        # positive = pr_probas[gt_probas > 0]
        # negative = pr_probas[gt_probas <= 0]
        # loss_bg = F.cross_entropy(
        #     positive,
        #     torch.ones_like(positive),
        # ) + F.cross_entropy(
        #     negative,
        #     torch.zeros_like(negative),
        # )
        # loss_bg = loss_bg / 2
        # pr_backgrounds = 1 - (pr_probas + pr_thresholds)
        # gt_backgrounds = 1 - (gt_probas + gt_thresholds)
        # loss_bg = F.cross_entropy(
        #     torch.stack(
        #         [pr_backgrounds, pr_probas, pr_thresholds],
        #         dim=1,
        #     ),
        #     torch.stack(
        #         [gt_backgrounds, gt_probas, gt_thresholds],
        #         dim=1,
        #     ).argmax(dim=1),
        # )

        # Proba map loss
        loss_proba = F.binary_cross_entropy_with_logits(
            pr_probas,
            gt_probas,
        )

        # Threshold map loss
        loss_threshold = F.l1_loss(pr_thresholds, gt_thresholds)

        # Binary map loss
        gt_binary = torch.sigmoid(1 * (gt_probas - gt_thresholds))
        pr_binary = torch.sigmoid(1 * (pr_probas - pr_thresholds))
        loss_binary = F.binary_cross_entropy(pr_binary, gt_binary)

        loss = loss_proba + loss_threshold + loss_binary
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

    def forward(self, features, tagets=None):
        outputs = self.outputs(features)
        return outputs

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

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

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth: bool = False):
        if ground_truth:
            outputs = outputs.unsqueeze(1)
        else:
            outputs = torch.clip(torch.sigmoid(50 * outputs), 0, 1)

        outputs = utils.stack_image_batch(outputs)

        logger.add_image(tag, outputs, step)
