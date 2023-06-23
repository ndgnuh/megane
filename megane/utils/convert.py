from PIL import Image
from io import BytesIO


def polygon2xyxy(polygon):
    xmin = min(x for x, y in polygon)
    xmax = max(x for x, y in polygon)
    ymin = min(y for x, y in polygon)
    ymax = max(y for x, y in polygon)
    return xmin, ymin, xmax, ymax


def xyxy2polygon(xyxy):
    xmin, ymin, xmax, ymax = xyxy
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def bytes2pillow(bs: bytes) -> Image:
    """
    Convert byte buffer to Pillow Image

    Args:
        bs: The byte buffer

    Returns:
        image: A PIL Image
    """
    io = BytesIO(bs)
    image = Image.open(io)
    image.load()
    io.close()
    return image
