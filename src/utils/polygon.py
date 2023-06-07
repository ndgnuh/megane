from typing import List, Tuple


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
