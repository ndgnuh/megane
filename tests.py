import numpy as np

from src import processors as P

def test_box_convert():
    xyxy = np.array([[10, 20, 300, 40], [0, 0, 100, 20]])
    points = P.xyxy_to_points(xyxy)

    assert np.all(points == np.array([
        [[10, 20], [300, 20], [300, 40], [10, 40]],
        [[0, 0], [100, 0], [100, 20], [0, 20]],
    ]))
    assert np.all(P.points_to_xyxy(points) == xyxy)
