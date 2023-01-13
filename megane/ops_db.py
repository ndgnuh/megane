from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2


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


def build_db_target(
    polygons: List[Tuple[Tuple[int]]],
    image_width: int,
    image_height: int,
    shrink_ratio: float = 0.4,
    min_thresh: float = 0.3,
    max_thresh: float = 0.7,
    min_box_size: int = 3,
):
    # Denormalize polygon
    polygons = polygons.copy()
    polygons[:, :, 0] *= image_width
    polygons[:, :, 1] *= image_height

    # Shrinks
    box_sizes = np.linalg.norm(polygons[:, 2, :] - polygons[:, 0, :], axis=-1)
    shrink_distances = np.array([
        (1 - shrink_ratio**2) * polygon_area(poly) / polygon_perimeter(poly)
        for poly in polygons
    ])
    shrinked_polygons = np.stack([
        offset_poly(polygon, -distance)
        for (polygon, distance) in zip(polygons, shrink_distances)
    ])
    expanded_polygons = np.stack([
        offset_poly(polygon, distance)
        for (polygon, distance) in zip(polygons, shrink_distances)
    ])

    # To int
    polygons = polygons.round().astype(int)
    shrinked_polygons = shrinked_polygons.round().astype(int)
    expanded_polygons = expanded_polygons.round().astype(int)

    # Draw maps and masks
    proba_map = np.zeros((image_height, image_width), dtype='float32')
    proba_mask = np.ones((image_height, image_width), dtype='float32')
    thresh_map = np.zeros((image_height, image_width), dtype='float32')
    thresh_mask = np.ones((image_height, image_width), dtype='float32')
    for polygon, shrink, expand in zip(polygons, shrinked_polygons, expanded_polygons):
        cv2.fillConvexPoly(proba_map, shrink, (1,))
        cv2.fillConvexPoly(proba_mask, polygon, (0,))
        cv2.fillConvexPoly(thresh_map, expand, (1,))
        cv2.fillConvexPoly(thresh_map, shrink, (0,))
        cv2.fillConvexPoly(thresh_mask, expand, (0,))

    # Distance map as threshold map
    thresh_map = cv2.distanceTransform(
        thresh_map,
        distanceType=cv2.DIST_L2,
        maskSize=3
    )
    # To a [0, 1] vector
    thresh_map = (thresh_map - thresh_map.min()) / \
        (thresh_map.max() - thresh_map.min())
    # To a [min thesh, max_thresh vector]
    thresh_map = (max_thresh - min_thresh) * thresh_map + min_thresh

    # Convert to uint8 255
    proba_map = (proba_map * 255).round().astype('uint8')
    proba_mask = (proba_mask * 255).round().astype('uint8')
    thresh_map = (thresh_map * 255).round().astype('uint8')
    thresh_mask = (thresh_mask * 255).round().astype('uint8')

    return (
        Image.fromarray(proba_map),
        Image.fromarray(proba_mask),
        Image.fromarray(thresh_map),
        Image.fromarray(thresh_mask),
    )
