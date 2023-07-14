import json
from base64 import b64decode
from dataclasses import dataclass, field
from functools import lru_cache
from os import path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from pyrsistent import pvector

from megane import utils


@dataclass
class Sample:
    """One text detection train data sample

    Attributes:
        image:
            The input Pillow image
        boxes:
            List of bounding box in the polygon format and
            has values normalized to (0, 1).
        classes:
            List of classes associated with each bounding boxes
        scores:
            Class scores (if available).
    """

    image: Image
    boxes: pvector = field(default_factory=pvector)
    classes: pvector = field(default_factory=pvector)
    scores: Optional[pvector] = None

    def __post_init__(self):
        # Validate
        self.image = self.image.convert("RGB")
        for box in self.boxes:
            box = np.array(box)
            assert (
                box.shape[-1] == 2 and box.ndim == 2
            ), f"Invalid bounding box format, {box.shape}"

    def visualize(self) -> Image:
        colors = [
            "#a54242",
            "#8c9440",
            "#de935f",
            "#5f819d",
            "#85678f",
            "#5e8d87",
            "#707880",
            "#cc6666",
            "#b5bd68",
            "#f0c674",
            "#81a2be",
            "#b294bb",
            "#8abeb7",
            "#c5c8c6",
        ]
        image = self.image.copy()
        w, h = image.size
        draw = ImageDraw.Draw(image)
        for polygon, class_id in zip(self.boxes, self.classes):
            xy = [(int(x * w), int(y * h)) for (x, y) in polygon]
            draw.polygon(xy, outline=colors[class_id], width=2)
        return image

    def visualize_tensor(self, *a, **k):
        from torchvision.transforms.functional import to_tensor

        image = self.visualize(*a, **k)
        return to_tensor(image)

    def adapt_metrics(self):
        import numpy as np

        boxes = np.array(self.boxes, dtype="object")
        classes = np.array(self.classes)
        return boxes, classes


def load_sample_labelme(sample_path, classes, single_class: bool):
    """Load a labelme json and convert it to an instance of Sample

    Args:
        sample_path:
            Path to sample labelme json.
        classes:
            List of class names, this is required to convert labelme
            class to number.

    Returns:
        sample:
            An instance of `Sample` corresponding to the labelme data.
    """
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    shapes = data["shapes"]
    image_path = data["imagePath"]
    image_data = data["imageData"]

    if image_data is not None:
        image = utils.bytes2pillow(b64decode(image_data))
    else:
        image_path = path.join(path.dirname(sample_path), image_path)
        image = Image.open(image_path)

    width, height = image.size

    boxes = pvector()
    class_indices = pvector()
    for shape in shapes:
        if shape["shape_type"] == "rectangle":
            [x1, y1], [x2, y2] = shape["points"]
            x1 = x1 / width
            y1 = y1 / height
            x2 = x2 / width
            y2 = y2 / height
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif shape["shape_type"] == "polygon":
            # xs = [x / width for (x, y) in shape["points"]]
            # ys = [y / height for (x, y) in shape["points"]]
            poly = [(x / width, y / height) for (x, y) in shape["points"]]
        else:
            continue
        boxes = boxes.append(poly)

        if single_class:
            class_indices = class_indices.append(0)
        else:
            assert shape["label"] in classes, f"Unknown class {shape['label']}"
            class_idx = classes.index(shape["label"])
            class_indices = class_indices.append(class_idx)

    assert len(class_indices) == len(boxes)

    return Sample(image=image, boxes=boxes, classes=class_indices)


class TextDetectionDataset(Dataset):
    """Text detection dataset

    Args:
        index:
            A text file contains path of all the labelme json files.
        classes:
            List of all class names.
        transform:
            A function that takes a `Sample` and returns another `Sample` somehow.
            The result of the transformation is returned when indexing the dataset.
        cache:
            Should the sample loading be cached?
            The transformation is still applied everytime.
        single_class:
            Should the class of data be ignored?
            If yes, everything will be put as positive and negative.
    """

    def __init__(
        self,
        index: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        cache: Optional[bool] = True,
        single_class: Optional[bool] = False,
    ):
        super().__init__()

        # Read sample paths
        root_dir = path.dirname(index)
        with open(index, "r") as fp:
            sample_paths = [line.strip("\n") for line in fp.readlines()]
            sample_paths = [
                path.join(root_dir, sample_path) for sample_path in sample_paths
            ]
        for sample_path in sample_paths:
            assert path.isfile(sample_path), f"{sample_path} does not exists"

        # Should the dataset be cached?
        if cache:
            _load_sample_labelme = lru_cache(load_sample_labelme)
        else:
            _load_sample_labelme = load_sample_labelme

        # Metadata
        self.index = index
        self.sample_paths = sample_paths
        self.transform = transform or (lambda x: x)
        self.classes = tuple(classes)
        self._load_sample = _load_sample_labelme
        self.single_class = single_class

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self._load_sample(
            self.sample_paths[index], self.classes, self.single_class
        )
        sample = self.transform(sample)
        return sample


def get_dataset(*a, **k):
    return TextDetectionDataset(*a, **k)
