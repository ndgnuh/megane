from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional, Tuple
from os import path, listdir
from PIL import Image
import json
from .tools import remember

IMG_EXTS = Image.registered_extensions()


def top_left_letterbox(image):
    width, height = image.size
    size = max(width, height)
    output = Image.new("RGB", (size, size), (0, 0, 0))
    output.paste(image, (0, 0))
    return output


def preprocess_image(image, width, height):
    # image.thumbnail((width, height), resample=Image.BILINEAR)
    # image = top_left_letterbox(image)
    image = image.resize((width, height), resample=Image.BILINEAR)
    return image


class EyeballDataset(Dataset):
    @remember
    def __init__(
        self,
        root: str,
        image_width: int,
        image_height: int,
        preprocess: Optional[Callable] = None,
        augment: Optional[Callable] = None
    ):
        super().__init__()
        self.samples = self.get_samples()

    def get_samples(self):
        files = listdir(self.root)
        samples = []

        for file in files:
            # Because there's only 1 annotation format
            basename, ext = path.splitext(file)
            if ext not in IMG_EXTS:
                continue

            annotation_file = path.join(self.root, f"{basename}.json")
            if not path.isfile(annotation_file):
                continue

            image_file = path.join(self.root, file)
            samples.append((image_file, annotation_file))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, annotation_file = self.samples[idx]
        image = Image.open(image_file)
        width, height = image.size
        image = preprocess_image(image, self.image_width, self.image_height)

        with open(annotation_file, encoding="utf-8") as f:
            annotation = json.load(f)
            # Since this is a detector dataset
            # No text information is needed
            annotation = annotation['boxes']

        # Augment comes first because
        # the outputs of preprocess is different
        if self.augment is not None:
            image, annotation = self.augment(image, annotation)

        if self.preprocess is not None:
            image, annotation = self.preprocess(image, annotation)

        return image, annotation
