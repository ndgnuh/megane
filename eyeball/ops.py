import json
import numpy as np
import cv2
from PIL import Image


def draw_rects(image, rects, **kwargs):
    from PIL import ImageDraw
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for rect in rects:
        draw.rectangle(rect, **kwargs)
    return result


def read_json(f):
    with open(f) as io:
        return json.load(io)


def top_left_letterbox(image):
    width, height = image.size
    size = max(width, height)
    output = Image.new("RGB", (size, size), (125, 125, 125))
    output.paste(image, (0, 0))
    return output


def xyxy_to_ccwh(box):
    x1, y1, x2, y2 = box
    w = (x2 - x1)
    h = (y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


def preprocess_image(image, width, height, letterbox=False):
    if letterbox:
        image = top_left_letterbox(image)
    # image.thumbnail((width, height), resample=Image.BILINEAR)
    image = image.resize((width, height), resample=Image.BILINEAR)
    return image


def normalize(box, width, height, norm_constant=1000):
    x1, y1, x2, y2 = box
    x1 = x1 * norm_constant / width
    y1 = y1 * norm_constant / height
    x2 = x2 * norm_constant / width
    y2 = y2 * norm_constant / height
    return x1, y1, x2, y2


def denormalize(box, width, height, norm_constant=1000):
    x1, y1, x2, y2 = box
    x1 = x1 / norm_constant * width
    y1 = y1 / norm_constant * height
    x2 = x2 / norm_constant * width
    y2 = y2 / norm_constant * height
    return x1, y1, x2, y2


def offset_rect(xyxy, r, expand=False):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1

    # Expand distance
    A = w * h
    L = 2 * (w + h)
    d = A * (1 - r**2) / L

    if expand:
        d = -d

    # Expanded
    x1 = max(x1 + d, 0)
    y1 = max(y1 + d, 0)
    x2 = x2 - d
    y2 = y2 - d
    return x1, y1, x2, y2


def mask_to_boxes(proba_map: np.ndarray,
                  min_box_size: int = 10,
                  min_box_score: int = 0.6,
                  expand_ratio: float = 1.5,
                  threshold: float = 0.02):
    mask = (255 * (proba_map > threshold)).astype('uint8')
    image_height, image_width = mask.shape

    # mask: h * w
    stats = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    boxes, scores = [], []
    for (x1, y1, w, h, s) in stats[2]:
        # Check for too-small boxes
        x2 = x1 + w
        y2 = y1 + h
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size or w >= 0.95 * image_width or h >= 0.95 * image_height:
            continue

        score = proba_map[y1:y2, x1:x2].mean()

        if np.isnan(score) or score <= min_box_score:
            continue

        box = (x1 / image_width, y1 / image_height,
               x2 / image_width, y2 / image_height)
        x1, y1, x2, y2 = offset_rect(box, expand_ratio)
        scores.append(score)
        boxes.append(box)

    return boxes, scores
