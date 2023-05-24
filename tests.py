import numpy as np
from PIL import Image

from src import processors as P

def test_box_convert():
    xyxy = np.array([[10, 20, 300, 40], [0, 0, 100, 20]])
    points = P.xyxy_to_points(xyxy)

    assert np.all(points == np.array([
        [[10, 20], [300, 20], [300, 40], [10, 40]],
        [[0, 0], [100, 0], [100, 20], [0, 20]],
    ]))
    assert np.all(P.points_to_xyxy(points) == xyxy)

def test_image_convert():
    data = np.random.randint(0, 255, (100, 200, 3), dtype='uint8')
    image = Image.fromarray(data)

    image_np = P.pil_to_np(image)
    assert image_np.shape == (3, image.height, image.width)

    image_pil = P.np_to_pil(image_np)
    assert (np.array(image_pil) == data).all()
