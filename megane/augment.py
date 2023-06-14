import albumentations as A
import numpy as np
from PIL import Image

from megane import utils
from megane.data import Sample


def encode(sample: Sample):
    """
    Encodes a Sample to albumentation inputs.

    Args:
        sample (Sample):
            The Sample object containing image and keypoints data.

    Returns:
        Dict:
            Albumentation input.

    Example:
        sample = Sample(image, boxes)
        enc = encode(sample)
        transform(**enc)
    """
    image = np.array(sample.image)
    w, h = sample.image.size
    boxes = utils.denormalize_polygon(sample.boxes, w, h, batch=True)
    masks = []
    keypoints = []
    for i, polygon in enumerate(boxes):
        keypoints.extend(polygon)
        masks.extend([i] * len(polygon))
    return dict(image=image, keypoints=keypoints, box_mapping=masks)


def decode(sample: Sample, outputs):
    """
    Decodes the albumenation outputs into a Sample object.

    Args:
        sample (Sample):
            The original Sample object.
        outputs (dict):
            The outputs obtained from the encoding process.

    Returns:
        Sample:
            The decoded Sample object.

    Example:
        sample = Sample(image, boxes, classes, scores)
        outputs = {"image": encoded_image,
            "box_mapping": box_mapping, "keypoints": keypoints}
        decoded_sample = decode(sample, outputs)
    """
    w, h = sample.image.size
    image = Image.fromarray(outputs["image"])

    # Convert keypoints to bounding boxes
    boxes = [[]] * len(sample.boxes)
    for mask, xy in zip(outputs["box_mapping"], outputs["keypoints"]):
        x, y = xy
        boxes[mask].append((x, y))

    # Remove invalid bounding boxes
    keeps = [len(box) > 2 for box in boxes]
    boxes = utils.normalize_polygon(boxes, w, h, batch=True)
    boxes = [c for (c, keep) in zip(boxes, keeps) if keep]
    classes = [c for (c, keep) in zip(sample.classes, keeps) if keep]
    if sample.scores is None:
        scores = None
    else:
        scores = [c for (c, keep) in zip(sample.scores, keeps) if keep]

    # Reconstruct another sample
    return Sample(
        image=image,
        boxes=boxes,
        classes=classes,
        scores=scores,
    )


class Augmentation:
    def __init__(self, p=0.5):
        self.transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=p),
                A.Rotate(10, p=p),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def __call__(self, sample: Sample) -> Sample:
        enc = encode(sample)
        enc = self.transform(enc)
        dec = decode(sample, enc)
        return dec
