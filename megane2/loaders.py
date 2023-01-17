import json
import os
from os import path
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class MeganeDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.samples = []
        for file in os.listdir(root):
            if not file.endswith(".json"):
                continue
            with open(path.join(root, file)) as f:
                self.samples.append(json.load(f))

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx):
        annotation = self.samples[idx]
        image = Image.open(path.join(self.root, annotation['image_path']))
        try:
            annotation.pop("width")
            annotation.pop("height")
        except Exception:
            pass
        return image, annotation

    def __getitem__(self, idx):
        image, annotation = self.load_sample(idx)
        if self.transform is not None:
            image, annotation = self.transform(image, annotation)
        return image, annotation
