import json
import warnings
from os import path
from typing import Dict, List, Tuple, Union

from PIL import Image
from pyrsistent import PVector, pvector
from torch.utils.data import DataLoader, Dataset

warnings.simplefilter("once", UserWarning)


def parse_labelme(
    sample_path: str,
    class_mapping: Union[List[str], Dict[str, int]],
) -> [Image.Image, PVector, PVector]:
    """Load a labelme json.

    Args:
        sample_path (str):
            Path to sample labelme json.
        class2str (Union[List[str], Dict[str, int]]):
            Either a list of class names, or a dictionary that maps
            class name to class index.
            This is required to convert labelme class to number.

    Returns:
        image (Pil.Image.Image):
            The image
        boxes (List[List[Tuple[int, int]]]):
            List of boxes in polygon format
        classes (List[int]):
            List of classes according to the box.
    """
    # Read data
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    root_path = path.dirname(sample_path)
    shapes = data["shapes"]
    image_path = data["imagePath"]

    # Load image
    image_path = path.join(root_path, image_path)
    image = Image.open(image_path)
    image.load()
    image_ = image.copy()
    width, height = image.size
    image.close()
    image = image_

    # The target boxes and classes
    boxes = pvector()
    classes = pvector()
    for shape in shapes:
        # Target boxes
        if shape["shape_type"] == "rectangle":
            [x1, y1], [x2, y2] = shape["points"]
            box = (x1 / width, y1 / height, x2 / width, y2 / height)
        elif shape["shape_type"] == "polygon":
            box = shape["points"]
            xs = [x[0] for x in box]
            ys = [x[1] for x in box]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            box = (x1 / width, y1 / height, x2 / width, y2 / height)
        else:
            continue

        # Target classes
        label = shape["label"]
        if label not in class_mapping:
            msg = f"Cannot find the label {label}, ignoring"
            warnings.warn(msg, UserWarning)
        else:
            if isinstance(class_mapping, list):
                class_idx = class_mapping.index(label)
            else:
                class_idx = class_mapping[label]
            classes = classes.append(class_idx)
            boxes = boxes.append(box)

    # Correctness check
    assert len(classes) == len(boxes)

    return image, boxes, classes


class LabelMeDataset(Dataset):
    def __init__(self, index_path: str, class_map, transform=None):
        super().__init__()
        root = path.dirname(index_path)
        with open(index_path) as f:
            sample_files = [line.strip() for line in f.readlines()]
            sample_files = [
                path.join(root, line) for line in sample_files if len(line) > 0
            ]
        self.root = root
        self.samples = sample_files
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, boxes, classes = parse_labelme(self.samples[idx], self.class_map)
        if self.transform is not None:
            return self.transform(image, boxes, classes)
        else:
            return image, boxes, classes
