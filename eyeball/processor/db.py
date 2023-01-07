from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2


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


def generate_db_masks(size, boxes, r=0.4):
    shrink = [offset_rect(box, r, False) for box in boxes]
    expand = [offset_rect(box, r, True) for box in boxes]
    proba_map = draw_mask_rect(size, shrink)
    threshold_map = draw_mask_rect(size, expand)
    threshold_map = ImageChops.subtract(threshold_map, proba_map)
    return proba_map, threshold_map


def mask_to_boxes(mask: np.ndarray,
                  min_box_size: int,
                  min_box_score: int,
                  expand_ratio: float,
                  proba_map: np.ndarray = None):
    # targets don't have proba map (maybe?)
    calculate_score = proba_map is not None
    image_height, image_width = mask.shape

    # mask: h * w
    stats = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    boxes, scores = [], []
    for (x1, y1, w, h, s) in stats[2]:
        # Check for too-small boxes
        x2 = x1 + w
        y2 = y1 + h
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue

        if calculate_score:
            score = proba_map[y1:y2, x1:x2].mean()
        else:
            score = 1

        if np.isnan(score) or score <= min_box_score:
            continue

        box = (x1 / image_width, y1 / image_height,
               x2 / image_width, y2 / image_height)
        x1, y1, x2, y2 = offset_rect(box, expand_ratio, True)
        scores.append(score)
        boxes.append(box)

    return boxes, scores


def polygon_area(poly):
    # https://en.wikipedia.org/wiki/Polygon#Area
    n = len(poly)
    area = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    area /= 2
    return area


def polygon_perimeter(poly):
    n = len(poly)
    peri = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        peri += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return peri


def offset_poly(poly, offset=1):
    n = len(poly)
    offset_lines = []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Calculate the direction vector
        vx, vy = x2 - x1, y2 - y1

        # flip direction vector by 90deg
        vx, vy = vy, -vx

        # normalize
        length = (vx**2 + vy**2)**0.5
        vx, vy = vx / length, vy / length

        # Offset line
        nx1 = x1 + vx * offset
        ny1 = y1 + vy * offset
        nx2 = x2 + vx * offset
        ny2 = y2 + vy * offset

        offset_lines.append((nx1, ny1, nx2, ny2))

    # New poly vertices are the intersection of the offset lines
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    new_poly = []
    for i in range(n):
        (x1, y1, x2, y2) = offset_lines[i]
        (x3, y3, x4, y4) = offset_lines[(i + 1) % n]
        deno = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
             (x1 - x2) * (x3 * y4 - x4 * y3)) / deno
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
             (y1 - y2) * (x3 * y4 - x4 * y3)) / deno
        new_poly.append((x, y))
    return new_poly


def draw_mask(size, polygons, fill=255):
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, fill=fill)
    return mask


def draw_mask_rect(size, rects, fill=255):
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    for rect in rects:
        draw.rectangle(rect, fill=fill)
    return mask


@dataclass
class DBProcessor:
    offset_ratio: float = 0.4
    expand_ratio: float = 1.5
    min_box_size: int = 10
    min_box_score: int = 0.6

    # I'm being lazy here
    # the code is adapted from a previous implementation using polygon
    def pre(self, image, boxes):
        from torchvision.transforms import functional as TF
        boxes = [denormalize(box, *image.size) for box in boxes]
        proba_map, threshold_map = generate_db_masks(
            image.size,
            boxes,
            self.offset_ratio
        )
        return (
            TF.to_tensor(image),
            (
                TF.to_tensor(proba_map),
                TF.to_tensor(threshold_map)
            )
        )

    def post(self, outputs, is_target=False):
        proba_maps, _ = outputs
        batch, nclasses, height, width = proba_maps.shape
        assert nclasses == 1
        if not is_target:
            import torch
            proba_maps = torch.sigmoid(proba_maps * 50)

        return [self.postprocess_single(pmap[0]) for pmap in proba_maps]

    def postprocess_single(self, proba_map):
        proba_map = proba_map.detach().cpu().numpy()
        boxes, scores = mask_to_boxes(
            mask=proba_map >= 0.5,
            min_box_size=self.min_box_size,
            min_box_score=self.min_box_score,
            expand_ratio=self.offset_ratio,
            proba_map=proba_map
        )
        return dict(boxes=boxes, scores=scores)
