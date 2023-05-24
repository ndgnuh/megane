import numpy as np


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
