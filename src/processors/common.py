import numpy as np
from PIL import Image


def xyxy_to_points(xyxy):
    """
    xyxy: n * 4
    """
    x1, y1, x2, y2 = xyxy.transpose([1, 0])
    tl = np.stack([x1, y1], axis=1)
    tr = np.stack([x2, y1], axis=1)
    br = np.stack([x2, y2], axis=1)
    bl = np.stack([x1, y2], axis=1)
    points = np.stack([tl, tr, br, bl], axis=1)
    return points


def points_to_xyxy(points):
    """
    points: n * 4 * 2
    """
    xmin, ymin = points.min(axis=1).transpose([1, 0])
    xmax, ymax = points.max(axis=1).transpose([1, 0])
    xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1)
    return xyxy


def pil_to_np(image: Image) -> np.ndarray:
    image_np = np.array(image)
    if image_np.dtype == "uint8":
        image_np = (image_np / 255).astype("float32")
        image_np = np.clip(image_np, 0, 1)

    h, w, c = 0, 1, 2
    image_np = image_np.transpose((c, h, w))
    return image_np


def np_to_pil(image_np: np.ndarray) -> Image:
    c, h, w = 0, 1, 2
    image = image_np.transpose([h, w, c])
    image = (image * 255).astype("uint8")
    image = Image.fromarray(image)
    return image


def normalize(points: np.ndarray, width: int, height: int):
    points = points * 1.0
    points[..., 0] /= width
    points[..., 1] /= height
    return points.astype('float32')


def denormalize(points: np.ndarray, width: int, height: int):
    points = points.copy()
    points[..., 0] *= width
    points[..., 1] *= height
    points = points.round().astype(int)
    return points
