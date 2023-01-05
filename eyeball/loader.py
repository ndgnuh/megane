from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional
from os import path, listdir
from PIL import Image
import json

IMG_EXTS = Image.registered_extensions()


class EyeballDataset(Dataset):
    def __init__(
        self,
        root: str,
        preprocess: Optional[Callable] = None,
        augment: Optional[Callable] = None
    ):
        super().__init__()
        self.root = root
        self.samples = self.get_samples()
        self.preprocess = preprocess
        self.augment = augment

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
