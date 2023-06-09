from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .polygon import polygon_area, polygon_perimeter, offset_polygon


def bytes2pillow(bs: bytes) -> Image:
    """
    Convert byte buffer to Pillow Image

    Args:
        bs: The byte buffer

    Returns:
        image: A PIL Image
    """
    io = BytesIO(bs)
    image = Image.open(io)
    image.load()
    io.close()
    return image


def prepare_input(image: Image, image_width: int, image_height: int):
    """Prepare the input to be fed to the model

    Args:
        image:
            Pillow image
        image_width:
            The image width W that model expects
        image_height:
            The image height H that model expects

    Returns:
        A numpy array of shape [3, H, W], type `float32`, value normalized to [0, 1] range.
    """
    image = image.convert("RGB")
    image = image.resize((image_width, image_height))
    image = np.array(image)
    image = image.astype("float32") / 255
    h, w, c = 0, 1, 2
    image = image.transpose([c, h, w])
    return image


def draw_mask(
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


def shrink(boxes: np.ndarray, r: float = 0.4):
    """Shrink bounding boxes using formular from DBNet.

    Shrink distance = (1 - r**2) * Area / Length

    Args:
        boxes:
            numpy array of shape [L, 4]
        r:
            Shrink ratio, default 0.4
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    length = (w + h) * 2
    d = (1.0 - r**2) * area / length
    new_boxes = np.stack(
        [boxes[:, 0] + d, boxes[:, 1] + d, boxes[:, 2] - d, boxes[:, 3] - d], axis=1
    )
    return new_boxes


def expand(boxes: np.ndarray, r: float = 1.5):
    """Expand bounding boxes using formular from DBNet.

    Expand distance = A * r / L

    Args:
        boxes:
            numpy array of shape [L, 4]
        r:
            Shrink ratio, default 0.4
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    length = (w + h) * 2
    d = 1.0 * r * area / length
    new_boxes = np.stack(
        [boxes[:, 0] - d, boxes[:, 1] - d, boxes[:, 2] + d, boxes[:, 3] + d], axis=1
    )
    return new_boxes


def shrink_expand(boxes: np.ndarray, r: float = 0.4):
    """Shrink and expand bounding boxes using the shrink formular from DBNet.

    Distance = (1 - r**2) * Area / Length

    Args:
        boxes:
            numpy array of shape [L, 4]
        r:
            Shrink ratio, default 0.4
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    length = (w + h) * 2
    d = (1.0 - r**2) * area / length
    shrinked = np.stack(
        [boxes[:, 0] + d, boxes[:, 1] + d, boxes[:, 2] - d, boxes[:, 3] - d], axis=1
    )
    expanded = np.stack(
        [boxes[:, 0] - d, boxes[:, 1] - d, boxes[:, 2] + d, boxes[:, 3] + d], axis=1
    )
    return shrinked, expanded


def mask_to_box(mask, min_score=0.5):
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
        scores[i] = np.clip(score_mask[y1:y2, x1:x2].mean(), 0, 1)

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
    return boxes, scores


def draw_mask_v2(width: int, height: int, polygons: np.ndarray):
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
        mask = cv2.fillConvexPoly(mask, polygon, 1)
    return mask


def mask_to_polygon(mask):
    """Convert from binary mask to box points polygon

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
    cnts, _ = cv2.findContours(
        (mask * 255).astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    def find_score(polygon):
        raster = np.zeros_like(mask, dtype="float32")
        raster = cv2.fillConvexPoly(raster, polygon.astype(int), 1)
        scores = mask * raster
        return scores.sum() / np.count_nonzero(scores)

    polygons = []
    scores = []
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        polygon = cv2.boxPoints(rect)
        score = find_score(polygon)
        polygons.append(polygon)
        scores.append(score)
    return polygons, scores


def shrink_polygon(polygon, r=0.4, A=None, L=None):
    """Shrink polygon using DB formula.
    Dist = (1 - r^2) * A / L

    Args:
        polygon:
            List of x y points
        r:
            Shrink ratio, default = 0.4
        A:
            Area of the polygon, will be computed if None is provided
        L:
            Length of the polygon, will be computed if None is provided

    Returns:
        The shrinked polygon.
    """
    A = A or polygon_area(poly)
    L = L or polygon_perimeter(poly)
    D = (1 - r**2) * A / L
    return offset_polygon(polygon, -D)


def expand_polygon(polygon, r=1.5, A=None, L=None):
    """Expand polygon using DB formula.
    Dist = r * A / L.

    Args:
        polygon:
            List of x y points
        r:
            Expand ratio, default = 1.5
        A:
            Area of the polygon, will be computed if None is provided
        L:
            Length of the polygon, will be computed if None is provided
    """
    A = A or polygon_area(poly)
    L = L or polygon_perimeter(poly)
    D = r * A / L
    return offset_polygon(polygon, D)
