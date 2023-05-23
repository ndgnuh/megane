from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import Sample


@dataclass
class Encoded:
    image: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    class_scores: np.ndarray
    box_scores: np.ndarray

    def to_tensor(self):
        import torch

        return Encoded(**{k: torch.tensor(v) for k, v in vars(self).items()})

    def to_batch(self):
        return Encoded(**{k: v.unqueeze(0) for k, v in vars(self).items()})

    def to_numpy(self):
        import torch

        return Encoded(**{k: v.cpu().detach().numpy() for k, v in vars(self).items()})


def encode(sample: Sample) -> Encoded:
    # encode image
    image = np.array(sample)
    if image.dtype == "uint8":
        image = (image / 255).astype("float32")
        image = np.clip(image, 0, 1)
    h, w, c = 0, 1, 2
    image = image.transpose((c, h, w))

    # encode bounding boxes
    return Encoded(
        image=image,
        boxes=np.array(sample.boxes),
        classes=np.array(sample.classes),
        class_scores=np.array(sample.class_scores),
        box_scores=np.array(sample.box_scores),
    )


def decode(enc: Encoded) -> Sample:
    # decode image
    c, h, w = 0, 1, 2
    image = image.transpose([h, w, c])
    image = Image.fromarray(image)
    return Sample(
        image=image,
        boxes=sample.boxes.tolist(),
        classes=sample.classes.tolist(),
        class_scores=sample.class_scores.tolist(),
        box_scores=sample.box_scores.tolist(),
    )
