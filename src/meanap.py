from typing import Tuple
import numpy as np


def torch_iou(b1, b2):
    """Wrapper for torchvision.ops.box_iou, receives and returns numpy array.
    This function is only used for testing
    """
    from torchvision import ops
    import torch

    b1 = torch.tensor(b1)
    b2 = torch.tensor(b2)
    return ops.box_iou(b1, b2).numpy()


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IOU between two set of boxes.
    Input boxes must be in [x1, y1, x2, y2] format.

    Args:
        boxes1:
            First set of boxes, numpy array of shape [L1, 4].
        boxes2:
            Second set of boxes, numpy array of shape [L2, 4].

    Returns:
        ious:
            IoU matrix of shape [L1, L2].
    """
    xmin1, ymin1, xmax1, ymax1 = boxes1.transpose(1, 0)
    xmin2, ymin2, xmax2, ymax2 = boxes2.transpose(1, 0)

    # Grid of intersect points
    x_near = np.max(np.meshgrid(xmin1, xmin2, indexing="ij"), axis=0)
    y_near = np.max(np.meshgrid(ymin1, ymin2, indexing="ij"), axis=0)
    x_far = np.min(np.meshgrid(xmax1, xmax2, indexing="ij"), axis=0)
    y_far = np.min(np.meshgrid(ymax1, ymax2, indexing="ij"), axis=0)

    # intersection
    intersection = (y_far - y_near) * (x_far - x_near)
    intersection = (intersection > 0) * intersection

    # union
    area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
    area2 = (ymax2 - ymin2) * (xmax2 - xmin2)
    union = area1[:, None] + area2[None, :] - intersection
    union = (union > 0) * union

    # IOU
    iou = intersection / union
    return iou


def compute_confusion(
    pr_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold=0.5
) -> Tuple[int, int, int]:
    """Compute confusion matrix, except for true negative

    Args:
        pr_boxes:
            Prediction bounding boxes.
            Shape: [L1, 4]
            Format: x1, y1, x2, y2
        gt_boxes:
            Ground truth bounding boxes.
            Shape: [L2, 4]
            Format: x1, y1, x2, y2
        iou_threshold:
            IoU score to be considered positive.

    Returns:
        tp:
            number of true positives
        fp:
            number of false positives
        fn:
            number of false negatives
    """
    num_pr = pr_boxes.shape[0]
    num_gt = gt_boxes.shape[0]

    # Corner cases
    if num_pr == 0:
        tp = fp = tn = 0
        fn = num_gt
        return tp, fp, fn
    if num_gt == 0:
        tp = fn = tn = 0
        fp = num_pr
        return tp, fp, fn

    # Find matches
    ious = compute_iou(pr_boxes, gt_boxes)
    pr_idx, gt_idx = np.where(ious >= iou_threshold)
    match_scores = ious[pr_idx, gt_idx]

    # No matches
    if match_scores.shape[0] == 0:
        tp = 0
        fp = num_pr
        fn = num_gt
        return tp, fp, fn

    # suppress matches
    args_desc = np.argsort(match_scores)[::-1]
    gt_match_idx = []
    pr_match_idx = []
    pr_unmatched = {}
    gt_unmatched = {}
    for idx in args_desc:
        _gt_idx = pr_idx[idx]
        _pr_idx = gt_idx[idx]
        if pr_unmatched.get(_pr_idx, True) and gt_unmatched.get(_gt_idx, True):
            pr_match_idx.append(_pr_idx)
            pr_unmatched[_pr_idx] = False
            gt_match_idx.append(_gt_idx)
            gt_unmatched[_gt_idx] = False
    tp = len(gt_match_idx)
    fp = num_pr - len(pr_match_idx)
    fn = num_gt - tp
    return tp, fp, fn


def compute_ap(
    pr_boxes: np.ndarray,
    pr_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    """Compute average precision of each classes.
    Boxes has [L, 4] shape, in xyxy format.
    Classes assignment has [L] shape.

    Args:
        pr_boxes:
            Prediction bounding boxes
        pr_classes:
            Prediction class assignment
        gt_boxes:
            Ground truth bounding boxes
        gt_classes:
            Ground truth class assignment

    Returns:
        AP numpy array of shape [C].
    """
    thresholds = np.linspace(0.0, 1.0, 10)
    num_threshold = len(thresholds)
    aps = []

    # Compute AP of each class
    for cls in np.unique(gt_classes):
        pr_mask = pr_classes == cls
        gt_mask = gt_classes == cls

        prs = []
        rcs = []
        for t in thresholds:
            tp, fp, fn = compute_confusion(
                pr_boxes[pr_mask], gt_boxes[gt_mask], iou_threshold=t
            )
            prs.append(tp / (tp + fp + 1e-6))
            rcs.append(tp / (tp + fn + 1e-6))
        ap = sum((rcs[i + 1] - rcs[i]) * prs[i + 1] for i in range(num_threshold - 1))
        aps.append(ap)

    return np.array(aps)


def compute_af1(
    pr_boxes: np.ndarray,
    pr_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    """Compute average F1 score of each class.
    Boxes has [L, 4] shape, in xyxy format.
    Classes assignment has [L] shape.

    Args:
        pr_boxes:
            Prediction bounding boxes
        pr_classes:
            Prediction class assignment
        gt_boxes:
            Ground truth bounding boxes
        gt_classes:
            Ground truth class assignment

    Returns:
        F1 numpy array of shape [C].
    """
    thresholds = np.linspace(0.0, 1.0, 10)
    num_threshold = len(thresholds)
    af1s = []

    # Compute AP of each class
    for cls in np.unique(gt_classes):
        pr_mask = pr_classes == cls
        gt_mask = gt_classes == cls

        f1s = []
        for t in thresholds:
            tp, fp, fn = compute_confusion(
                pr_boxes[pr_mask], gt_boxes[gt_mask], iou_threshold=t
            )
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            f1s.append(f1)
        af1s.append(np.mean(f1s))

    return np.array(af1s)


def compute_map(
    pr_boxes: np.ndarray,
    pr_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> float:
    """Compute average meanAP score.
    Boxes has [L, 4] shape, in xyxy format.
    Classes assignment has [L] shape.

    Args:
        pr_boxes:
            Prediction bounding boxes
        pr_classes:
            Prediction class assignment
        gt_boxes:
            Ground truth bounding boxes
        gt_classes:
            Ground truth class assignment

    Returns:
        Mean AP score.
    """
    return compute_ap(pr_boxes, pr_classes, gt_boxes, gt_classes).mean()


def compute_maf1(
    pr_boxes: np.ndarray,
    pr_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> float:
    """Compute mean of average F1 score over classes.
    Boxes has [L, 4] shape, in xyxy format.
    Classes assignment has [L] shape.

    Args:
        pr_boxes:
            Prediction bounding boxes
        pr_classes:
            Prediction class assignment
        gt_boxes:
            Ground truth bounding boxes
        gt_classes:
            Ground truth class assignment

    Returns:
        Mean average F1 score.
    """
    return compute_af1(pr_boxes, pr_classes, gt_boxes, gt_classes).mean()
