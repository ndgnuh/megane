from io import BytesIO
from PIL import Image


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
