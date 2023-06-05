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


def compute_iou(boxes1, boxes2):
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
