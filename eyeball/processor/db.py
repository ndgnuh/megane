from dataclasses import dataclass, field
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import numpy as np


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


@dataclass
class DBProcessor:
    expand_ratio: float = 0.4

    # I'm being lazy here
    # the code is adapted from a previous implementation using polygon
    def pre(self, image, boxes):
        width, height = image.size

        prob_polygons = []
        outer_polygons = []
        inner_polygons = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            x1 = x1 / 1000 * width
            x2 = x2 / 1000 * width
            y1 = y1 / 1000 * height
            y2 = y2 / 1000 * height

            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height
            length = (box_width + box_height) * 2
            distance = area * (1 - self.expand_ratio ** 2) / length

            polygon = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
            outer_polygon = offset_poly(polygon, distance)
            inner_polygon = offset_poly(polygon, -distance)

            prob_polygons.append(polygon)
            outer_polygons.append(outer_polygon)
            inner_polygons.append(inner_polygon)

        # Create masks
        prob_map = draw_mask((width, height), prob_polygons)
        ext_map = draw_mask((width, height), outer_polygons)
        shr_map = draw_mask((width, height), inner_polygons)
        thres_map = Image.fromarray(
            np.array(ext_map) - np.array(shr_map)
        )

        return (
            TF.to_tensor(image),
            (
                TF.to_tensor(shr_map),
                TF.to_tensor(thres_map)
            )
        )
