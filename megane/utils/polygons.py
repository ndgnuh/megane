from typing import List, Tuple


def normalize_polygon(polygon, width, height, batch=False):
    """
    Normalizes a polygon by scaling its coordinates to the range [0, 1].

    Args:
        polygon (list):
            The polygon to be normalized, represented as a list of vertices.
        width (int):
            The width of the bounding box containing the polygon.
        height (int):
            The height of the bounding box containing the polygon.
        batch (bool, optional):
            Specifies whether the input is a batch of polygons.
            Defaults to False.

    Returns:
        list:
            The normalized polygon, represented as a list of vertices.
            If batch is specified, the list contain multiple polygons.

    Example:
        polygon = [(10, 20), (30, 40), (50, 60)]
        normalized_polygon = normalize_polygon(polygon, 100, 200)
    """
    if batch:
        return [normalize_polygon(p, width, height) for p in polygon]
    return [(x / width, y / height) for x, y in polygon]


def denormalize_polygon(polygon, width, height, batch=False):
    """
    Denormalizes a polygon by scaling its coordinates from the range [0, 1]
    to the original width and height.

    Args:
        polygon (list):
            The polygon to be denormalized, represented as a list of vertices.
        width (int):
            The original width of the bounding box containing the polygon.
        height (int):
            The original height of the bounding box containing the polygon.
        batch (bool, optional):
            Specifies whether the polygon is part of a batch.
            Defaults to False.

    Returns:
        list:
            The normalized polygon, represented as a list of vertices.
            If batch is specified, the list contain multiple polygons.

    Example:
        polygon = [(0.2, 0.4), (0.6, 0.8), (1.0, 1.2)]
        denormalized_polygon = denormalize_polygon(polygon, 100, 200)
    """
    if batch:
        return [denormalize_polygon(p, width, height) for p in polygon]
    return [(int(x * width), int(y * height)) for x, y in polygon]


def polygon_area(poly: List[Tuple[float, float]]):
    """Calculate area of a polygon.

    Args:
        poly:
            List of tuple representing x, y points.
            Numpy arrays of shape [P, 2] would do too.

    References:
        https://en.wikipedia.org/wiki/Polygon#Area
    """
    n = len(poly)
    area = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area /= 2
    return area


def polygon_perimeter(poly: List[Tuple[float, float]]):
    """Calculate the perimeter of a polygon.

    Args:
        poly:
            List of tuple representing x, y points.
            Numpy arrays of shape [P, 2] would do too.

    References:
        https://en.wikipedia.org/wiki/Polygon#Area
    """
    n = len(poly)
    peri = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        peri += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return peri


def offset_polygon(poly: List[Tuple[float, float]], offset: float):
    """Offset the polygon by a some value.
    Negative offset shrink the polygon. Positive offset expand the polygon.

    Args:
        poly:
            List of (x, y) points. Numpy arrays of shape [P, 2] would do too.
        offset:
            Offset value.

    Refs:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
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
        length = (vx**2 + vy**2) ** 0.5 + 1e-6
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
        deno = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-6
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - x4 * y3)) / deno
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - x4 * y3)) / deno
        new_poly.append((x, y))
    return new_poly


def offset_polygon_np(p, d):
    """This version is slower somehow... Do not use"""
    import numpy as np

    starts = p
    ends = np.roll(p, -1, axis=0)

    # Finding the normal of each lines
    n = np.stack([starts[..., 1] - ends[..., 1], ends[..., 0] - starts[..., 0]], axis=1)
    n = n / np.linalg.norm(n, axis=-1)[..., None]

    # Shift the lines
    # print(n.shape, np.linalg.norm(n,).shape)
    shift = n * d
    starts = starts + shift
    ends = ends + shift

    # Finding shifted line intersections
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    starts1 = starts
    ends1 = ends
    starts2 = np.roll(starts1, 1, axis=0)
    ends2 = np.roll(ends1, 1, axis=0)

    n1 = starts1 - ends1
    n2 = starts2 - ends2
    # print(n1.shape, np.linalg.norm(n1, axis=1).shape)
    n1 = n1 / np.linalg.norm(n1, axis=1)[..., None]
    n2 = n2 / np.linalg.norm(n2, axis=1)[..., None]

    I = np.eye(2)[None, ...]
    A1 = I - np.matmul(np.expand_dims(n1, -1), np.expand_dims(n1, -2))
    A2 = I - np.matmul(n2[..., None], n2[..., None, :])
    C = np.matmul(A1, ends1[..., None]) + np.matmul(A2, ends2[..., None])
    S = np.stack([np.linalg.pinv(s) for s in A1 + A2], axis=0)
    new_polygons = np.matmul(S, C).squeeze(-1)
    return new_polygons
