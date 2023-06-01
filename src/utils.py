from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image


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
    image = image.astype('float32') / 255
    h, w, c = 0, 1, 2
    image = image.transpose([c, h, w])
    return image


def draw_mask(width: int, height: int, boxes: np.ndarray, copy: bool):
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
    Returns:
        A numpy array of shape [H, W] which is the drawn mask.
    """
    mask = np.zeros((height, width), dtype=int)
    if copy:
        boxes = boxes.copy()
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes = boxes.round().astype(int)

    for (x1, y1, x2, y2) in boxes:
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
    d = (1 - r**2) * area / length
    new_boxes = np.stack([
        boxes[:, 0] + d,
        boxes[:, 1] + d,
        boxes[:, 2] - d,
        boxes[:, 3] - d
    ], axis=1)
    return new_boxes
