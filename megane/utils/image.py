import numpy as np
from PIL import Image


def letterbox(
    image: Image,
    width: int,
    height: int,
    fill_value: int = 0,
) -> Image:
    """
    Resize and pad an image to fit within a specified
    width and height while maintaining the aspect ratio.

    Args:
        img:
            The input Pillow Image
        width:
            The desired width of the letterboxed image.
        height:
            The desired height of the letterboxed image.
        fill_value:
            The value to fill the padding with. Default is 0.

    Returns:
        The letterboxed image as a Pillow Image.
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Resize the image while maintaining the aspect ratio
    if aspect_ratio > width / height:
        new_width = width
        new_height = int(width / aspect_ratio)
    else:
        new_width = int(height * aspect_ratio)
        new_height = height

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a blank canvas of the specified width and height
    letterboxed_image = Image.new("RGB", (width, height), fill_value)

    # Calculate the padding for both width and height
    pad_width = (width - new_width) // 2
    pad_height = (height - new_height) // 2

    # Paste the resized image onto the letterboxed canvas
    letterboxed_image.paste(resized_image, (pad_width, pad_height))

    return letterboxed_image


def pillow_to_numpy(image: Image) -> np.ndarray:
    """
    Converts a PIL Image object to a RGB numpy array with CHW format.

    Args:
        image (Image): The input PIL Image object.

    Returns:
        np.ndarray: The numpy array representation of the image.
    """
    img = np.array(image.convert("RGB"), dtype="float32")
    img = img / 255
    h, w, c = 0, 1, 2
    img = img.transpose(c, h, w)
    return img


def prepare_input(
    image: Image,
    image_width: int,
    image_height: int,
    resize_mode: str = "resize",
    center_value: bool = False,
):
    """Prepare the input to be fed to the model

    Args:
        image:
            Pillow image
        image_width:
            The image width W that model expects
        image_height:
            The image height H that model expects
        resize_mode:
            Either "resize" or "letterbox"
        center_value:
            Normalize pixel values to [-1, 1] range. Default to false.

    Returns:
        A numpy array of shape [3, H, W], type `float32`, value normalized to [0, 1] range.
    """
    image = image.convert("RGB")
    if resize_mode == "resize":
        image = image.resize((image_width, image_height))
    elif resize_mode == "letterbox":
        image = letterbox(image, image_width, image_height)
    image = np.array(image)
    image = image.astype("float32") / 255
    h, w, c = 0, 1, 2
    image = image.transpose([c, h, w])
    if center_value:
        image = image * 2 - 1
    return image
