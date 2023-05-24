import numpy as np
from PIL import Image

from src import processors as P


def test_box_convert():
    xyxy = np.array([[10, 20, 300, 40], [0, 0, 100, 20]])
    test_points = [
        [[10, 20], [300, 20], [300, 40], [10, 40]],
        [[0, 0], [100, 0], [100, 20], [0, 20]],
    ]
    points = P.xyxy_to_points(xyxy)

    assert np.all(points == np.array(test_points))
    assert np.all(P.points_to_xyxy(points) == xyxy)


def test_image_convert():
    data = np.random.randint(0, 255, (100, 200, 3), dtype="uint8")
    image = Image.fromarray(data)

    image_np = P.pil_to_np(image)
    assert image_np.shape == (3, image.height, image.width)

    image_pil = P.np_to_pil(image_np)
    assert (np.array(image_pil) == data).all()


def test_box_normalization():
    width, height, n = 100, 200, 80
    x = np.random.randint(0, width, (n, 4))
    y = np.random.randint(0, height, (n, 4))
    points = np.stack([x, y], axis=2)

    normalized = P.normalize(points, width, height)
    assert np.all(normalized <= 1)
    assert np.all(normalized >= 0)

    denormalized = P.denormalize(normalized, width, height)
    assert np.all(denormalized - points < 0.1)
