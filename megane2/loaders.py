import json
import os
from os import path
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from functools import lru_cache


class IndexedImageFolder(VisionDataset):
    def __init__(self,
                 index: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 index_encoding: str = 'utf-8',
                 **kwargs):
        root = path.dirname(index)
        super().__init__(
            root=root,
            transforms=transforms
        )

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.index = index
        with open(self.index, encoding=index_encoding) as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [path.join(root, line) for line in lines if len(line) > 0]
        self.samples = lines

    def load_sample(self, sample: str):
        raise NotImplementedError("Error")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image, target = self.load_sample(self.samples[index])
        if self.transforms is not None:
            image = self.transforms(image)
            target = self.transforms(target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class MeganeDatasetOld(Dataset):
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
        image = Image.open(
            path.join(self.root, annotation['image_path'])
        ).convert("RGB")
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


class MeganeDataset(IndexedImageFolder):
    @lru_cache
    def get_annotation(self, idx):
        with open(self.samples[idx], encoding="utf-8") as io:
            data = json.load(io)
        return data

    def load_sample(self, idx):
        annotation = self.get_annotation(idx)
        root = path.dirname(self.samples[idx])
        image = Image.open(path.join(root, annotation['image_path']))
        image = image.convert("RGB")
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


def megane_dataloader(
    root: str,
    transform: Optional[Callable] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: Optional[int] = 0
):
    dataset = MeganeDataset(root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
