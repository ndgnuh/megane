import json
from base64 import b64decode
from dataclasses import dataclass, field
from functools import lru_cache
from os import path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

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
    boxes: List[List[Tuple[float, float]]] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    scores: Optional[List[float]] = None

    def __post_init__(self):
        # Validate
        for box in self.boxes:
            box = np.array(box)
            assert (
                box.shape[-1] == 2 and box.ndim == 2
            ), f"Invalid bounding box format, {box.shape}"

    def visualize(self, outline=(255, 0, 0)) -> Image:
        image = self.image.copy()
        w, h = image.size
        draw = ImageDraw.Draw(image)
        for polygon in self.boxes:
            xy = [(int(x * w), int(y * h)) for (x, y) in polygon]
            draw.polygon(xy, outline=outline)
        return image

    def visualize_tensor(self, *a, **k):
        from torchvision.transforms.functional import to_tensor

        image = self.visualize(*a, **k)
        return to_tensor(image)

    def adapt_metrics(self):
        import numpy as np

        boxes = np.array(self.boxes)
        classes = np.array(self.classes)
        return boxes, classes


def load_sample_labelme(sample_path, classes):
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
    width = data["imageWidth"]
    height = data["imageHeight"]
    image_path = data["imagePath"]
    image_data = data["imageData"]

    boxes = []
    class_indices = []
    for shape in shapes:
        if shape["shape_type"] == "rectangle":
            [x1, y1], [x2, y2] = shape["points"]
            x1 = x1 / width
            y1 = y1 / height
            x2 = x2 / width
            y2 = y2 / height
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            boxes.append(poly)
        elif shape["shape_type"] == "polygon":
            # xs = [x / width for (x, y) in shape["points"]]
            # ys = [y / height for (x, y) in shape["points"]]
            poly = [(x / width, y / height) for (x, y) in shape["points"]]
            boxes.append(poly)
        else:
            continue

        class_idx = classes.index(shape["label"])
        class_indices.append(class_idx)

    assert len(class_indices) == len(boxes)

    if image_data is not None:
        image = utils.bytes2pillow(b64decode(image_data))
    else:
        image_path = path.join(path.dirname(sample_path), image_path)
        image = Image.open(image_path)
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
    """

    def __init__(
        self,
        index: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        cache: Optional[bool] = True,
    ):
        super().__init__()
        root_dir = path.dirname(index)
        with open(index, "r") as fp:
            sample_paths = [line.strip("\n") for line in fp.readlines()]
            sample_paths = [
                path.join(root_dir, sample_path) for sample_path in sample_paths
            ]
        for sample_path in sample_paths:
            assert path.isfile(sample_path), f"{sample_path} does not exists"
        if cache:
            _load_sample_labelme = lru_cache(load_sample_labelme)
        else:
            _load_sample_labelme = load_sample_labelme
        self.index = index
        self.sample_paths = sample_paths
        self.transform = transform or (lambda x: x)
        self.classes = tuple(classes)
        self._load_sample = _load_sample_labelme

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self._load_sample(self.sample_paths[index], self.classes)
        sample = self.transform(sample)
        return sample
