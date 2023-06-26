import math
from typing import List

import cv2
import numpy as np
from megane.utils.convert import xyxy2polygon


def smooth(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    blur = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    remask = blur > 0.2
    return remask


def find_score(mask, polygon):
    """Calculates the score by computing the intersection between
    a binary mask and a polygon.

    The function fills the polygon with ones in a raster image and performs
    element-wise multiplication between the mask and the raster to obtain
    the intersection. The score is then calculated as the sum of the
    intersection divided by the count of non-zero elements in the intersection.

    Args:
        mask:
            Score mask
        polygon:
            An array representing the vertices of the polygon.

    Returns:
        The score.

    Example:
        >>> mask = np.array([[0, 1, 1],
        ...                  [0, 1, 0],
        ...                  [1, 0, 1]], dtype=np.uint8)
        >>> polygon = np.array([[1, 0],
        ...                     [2, 0],
        ...                     [2, 1]], dtype=np.float32)
        >>> score = find_score(mask, polygon)
    """
    raster = np.zeros_like(mask, dtype="float32")
    raster = cv2.fillPoly(raster, [polygon.astype(int)], 1)
    scores = mask * raster
    return scores.sum() / np.count_nonzero(scores)


def draw_threshold_mask(
    width: int,
    height: int,
    inner_polygons: List,
    outer_polygons: List,
):
    """
    Draws a threshold mask based on the provided inner and outer polygons.

    Args:
        width (int):
            The width of the mask.
        height (int):
            The height of the mask.
        inner_polygons (List):
            A list of inner polygons represented as a list of vertices.
        outer_polygons (List):
            A list of outer polygons represented as a list of vertices.

    Returns:
        np.ndarray:
            The generated threshold mask as a numpy array.

    Example:
        inner_polygons = [[[0, 0], [0, 5], [5, 5], [5, 0]]]
        outer_polygons = [[[0, 0], [0, 10], [10, 10], [10, 0]]]
        mask = draw_threshold_mask(10, 10, inner_polygons, outer_polygons)
    """
    mask = np.zeros((height, width), dtype="float32")
    for inner_box, outer_box in zip(inner_polygons, outer_polygons):
        # Draw to a canvas first
        # and then fill the inner box with background
        canvas = np.zeros_like(mask)
        canvas = cv2.fillPoly(canvas, [np.array(outer_box).astype(int)], 1, cv2.LINE_AA)
        canvas = cv2.fillPoly(canvas, [np.array(inner_box).astype(int)], 0, cv2.LINE_AA)
        # yank the canvas to the threshold map
        mask = mask + canvas
    # Normalize threshold map to 0..1
    mask = np.clip(mask, 0, 1)
    return mask


def draw_mask(width: int, height: int, polygons: np.ndarray):
    """Draw binary mask using polygon boxes.
    The mask region has value 1 whenever there's a box in that region.

    Args:
        width:
            The mask width.
        height:
            The mask height.
        boxes:
            A list of polygons (array of shape [L, 2]).
    Returns:
        A numpy array of shape [H, W] which is the drawn mask.
    """
    # mask_to_box
    mask = np.zeros((height, width), dtype="float32")
    for polygon in polygons:
        mask = cv2.fillPoly(mask, [np.array(polygon).astype(int)], 1, cv2.LINE_AA)
    return mask


def mask_to_rrect(mask, open_kernel=None):
    """Convert from binary mask to rotated rectangles

    Args:
        mask:
            2D numpy array, value in range 0, 1

    Returns:
        polygons:
            List of polygons.
        scores:
            Polygon scores based on the input mask.
    """
    height, width = mask.shape

    if open_kernel is not None:
        kernel = np.ones(open_kernel)
        bin_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        bin_mask = (mask > 0.5).astype("float32")
    bin_mask = (bin_mask * 255).astype("uint8")
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def should_square(rect):
        (x, y), (w, h), a = rect
        a = abs(a) % 90
        a = math.radians(a)
        r = abs(w / h - 1)
        return min(r, math.sin(a), math.cos(a)) < 0.16  # This is sin(20deg)

    polygons = []
    scores = []
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), a = rect
        if h == 0 or w == 0:
            continue
        polygon = cv2.boxPoints(rect)
        score = find_score(mask, polygon)
        if should_square(rect):
            xmin, ymin = np.min(polygon, axis=-2)
            xmax, ymax = np.max(polygon, axis=-2)
            polygon = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        polygons.append(polygon)
        scores.append(score)
    return polygons, scores


def mask_to_polygons(mask):
    """Convert from binary mask to rotated rectangles

    Args:
        mask:
            2D numpy array, value in range 0, 1

    Returns:
        polygons:
            List of polygons.
        scores:
            Polygon scores based on the input mask.
    """
    mask = smooth(mask)
    height, width = mask.shape
    bin_mask = mask > 0.5
    bin_mask = (bin_mask * 255).astype("uint8")
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    scores = []
    for cnt in cnts:
        cnt = cv2.convexHull(cnt)
        eps = cv2.arcLength(cnt, True) * 0.05
        # Do this to avoid rounded corners
        cnt = cv2.approxPolyDP(cnt, eps, closed=False)
        cnt = cv2.approxPolyDP(cnt, eps, closed=True)
        if cnt.shape[0] < 3:
            continue

        polygon = cnt[:, 0, :]
        score = find_score(mask, polygon)
        polygon = [(x / width, y / height) for (x, y) in polygon]
        polygons.append(polygon)
        scores.append(score)
    return polygons, scores


def draw_mask_v1(
    width: int, height: int, boxes: np.ndarray, copy: bool, mode: str = "max"
):
    """Draw binary mask using bounding boxes.
    The mask region has value 1 whenever there's a box in that region.

    Args:
        width:
            The mask width.
        height:
            The mask height.
        boxes:
            A numpy array reprensenting normalized bounding boxes of shape [L, 4].
        copy:
            Whether to copy when de-normalizing the boxes.
        mode:
            Either 'max', 'min' or 'round'. Mode max will try to maximize the box area.
            Round would not try to do anything. Min will try to minimize the box area.
            Default: 'max'
    Returns:
        A numpy array of shape [H, W] which is the drawn mask.
    """
    mask = np.zeros((height, width), dtype=int)
    if copy:
        boxes = boxes.copy()
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height

    # Bound rounding
    if mode == "max":
        boxes[:, [0, 1]] = np.floor(boxes[:, [0, 1]])
        boxes[:, [2, 3]] = np.ceil(boxes[:, [2, 3]])
        boxes = boxes.astype(int)
    elif mode == "min":
        boxes[:, [0, 1]] = np.ceil(boxes[:, [0, 1]])
        boxes[:, [2, 3]] = np.floor(boxes[:, [2, 3]])
        boxes = boxes.astype(int)
    elif mode == "round":
        boxes = boxes.round()
    else:
        raise ValueError(f"Unsupported mode {mode}")

    boxes = boxes.astype(int)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask


def mask_to_rect(mask, min_score=0.5):
    """Convert a score mask to bounding boxes using connected component.

    Args:
        mask:
            A numpy array score mask of shape [H, W], type float32.
        min_score:
            Min score filter, boxes with lower score would be filtered out.
            Default: 0.5

    Returns:
        boxes:
            A numpy float32 array of shape [L, 4], each row is a box of xyxy format.
            The boxes are normalized to 0..1 value range.
        scores:
            A numpy vector of length [L] that contains box scores.
            Each score is in 0..1 range.
    """
    # mask is float32
    imask = np.tanh(mask)
    imask = np.clip(imask, 0, 1) * 255
    imask = imask.round().astype("uint8")

    # Find boxes
    stats = cv2.connectedComponentsWithStats(imask)
    boxes = stats[2][1:, :4].copy()

    # Convert from xywh to xyxy
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # Calculate scores
    scores = np.zeros(boxes.shape[0])
    M = mask.max()
    m = mask.min()
    score_mask = (mask - m) / (M - m + 1e-6)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        aux = score_mask[y1:y2, x1:x2]
        score_nz = aux[aux > 0]
        scores[i] = np.clip(score_nz.mean(), 0, 1)

    # Expand boxes
    boxes = boxes.round().astype(int)

    # Normalize
    h, w = mask.shape
    normalizer = np.array([w, h, w, h]).reshape(1, 4)
    boxes = boxes / normalizer

    # Filter low score boxes
    keep = scores >= min_score
    boxes = boxes[keep, :]
    scores = scores[keep]

    # Convert box to polygon format
    boxes = [xyxy2polygon(xyxy) for xyxy in boxes]
    return boxes, scores
