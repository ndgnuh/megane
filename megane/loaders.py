import json
import os
import numpy as np
from os import path
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from functools import lru_cache
from base64 import b64decode
from tqdm import tqdm
from io import BytesIO
from scipy import io as sio


def create_lmdb_dataset(root, dataset):
    import lmdb
    env = lmdb.open(root, map_size=int(1e13))
    with env.begin(write=True) as txn:
        count = 0
        for image, annotation in tqdm(dataset, f"Writing to {root}"):
            image_bin = image_to_bytes(image)
            annotation_bin = json.dumps(annotation).encode()
            # labels_bin = numpy_to_bytes(np.array(annotation['labels']))
            # polygons_bin = numpy_to_bytes(np.array(annotation['polygons']))
            txn.put(f"image_{count:09d}".encode(), image_bin)
            txn.put(f"annotation_{count:09d}".encode(), annotation_bin)
            # txn.put(f"labels_{count:09d}".encode(), polygons_bin)
            count = count + 1

        txn.put("num_samples".encode(), str(count).encode())
    env.close()


def numpy_to_bytes(x: np.ndarray):
    io = BytesIO()
    np.save(io, x)
    return io.getvalue()


def numpy_from_bytes(bs):
    io = BytesIO(bs)
    return np.load(io)


def image_from_bytes(bs):
    io = BytesIO(bs)
    return Image.open(io)


def image_to_bytes(image):
    io = BytesIO()
    image.save(io, "PNG")
    bs = io.getbuffer()
    return bs


def load_sample_synthtext(sample: tuple):
    images, polygons, idx = sample
    image = Image.open(images[idx])
    polygons = polygons[idx].transpose([2, 1, 0]).copy()
    polygons[:, :, 0] = polygons[:, :, 0] / image.width
    polygons[:, :, 1] = polygons[:, :, 1] / image.height
    annotation = dict(
        polygons=polygons.tolist(),
        labels=[0] * polygons.shape[0]
    )
    return image, annotation


def load_sample_lmdb(sample: tuple):
    env, idx = sample
    with env.begin() as txn:
        image = image_from_bytes(txn.get(f"image_{idx:09d}".encode()))
        annotation = txn.get(f"annotation_{idx:09d}".encode())

    annotation = json.loads(annotation.decode("utf-8"))

    return image, annotation


def load_sample_megane(sample):
    with open(sample, encoding="utf-8") as f:
        annotation = json.load(f)
    root = path.dirname(sample)

    image = Image.open(path.join(root, annotation['imagePath']))
    image = image.convert("RGB")
    try:
        annotation.pop("width")
        annotation.pop("height")
    except Exception:
        pass
    return image, annotation


def load_sample_labelme(sample):
    with open(sample, encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes']
    width = data['imageWidth']
    height = data['imageHeight']
    image_path = data['imagePath']
    image_data = data['imageData']

    polygons = []
    labels = []
    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            [x1, y1], [x2, y2] = shape['points']
            polygon = [
                (x1 / width, y1 / height),
                (x2 / width, y1 / height),
                (x2 / width, y2 / height),
                (x1 / width, y2 / height),
            ]
        elif shape['shape_type'] == 'polygon':
            polygon = [
                (x / width, y / height)
                for (x, y) in shape['points']
            ]

        labels.append(shape['label'])
        polygons.append(polygon)
    label_maps = sorted(list(set(list(labels))))
    labels = [label_maps.index(label) for label in labels]

    annotation = dict(
        polygons=polygons,
        labels=labels,
        image_path=image_path,
    )
    if image_data is not None:
        image = image_from_bytes(b64decode(image_data))
    else:
        image_path = path.join(path.dirname(file), image_path)
        image = Image.open(image_path)
    return image, annotation


class MeganeDataset(Dataset):
    def __init__(self,
                 index: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 index_encoding: str = 'utf-8',
                 **kwargs):
        super().__init__()
        root = path.dirname(index)

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.index = index
        self.index_encoding = index_encoding
        self.samples = self.read_index(index)
        self.load_sample = self.detect_sample_loader()
        self.max_image_size = os.environ.get("MAX_IMAGE_SIZE", 1280)
        print(self.load_sample)

    def __len__(self):
        return len(self.samples)

    def read_index(self, index):
        if index.endswith(".txt"):
            with open(self.index, encoding=self.index_encoding) as f:
                lines = [line.strip() for line in f.readlines()]
                lines = [path.join(self.root, line)
                         for line in lines if len(line) > 0]
            return lines
        elif index.endswith(".mdb"):
            import lmdb
            env = lmdb.open(path.dirname(self.index),
                            lock=False,
                            map_size=int(1e13))
            with env.begin() as txn:
                num_samples = txn.get("num_samples".encode())
                num_samples = int(num_samples.decode())
            samples = [(env, count) for count in range(num_samples)]
            return samples
        elif index.endswith(".mat"):
            # SynthText
            import scipy.io as sio
            root = path.dirname(index)
            gt = sio.loadmat(index)
            images = [path.join(root, image[0])
                      for image in gt['imnames'].flatten()]
            bboxes = gt['charBB'][:].flatten()
            total = gt['imnames'][:].shape[-1]
            return [(images, bboxes, i) for i in range(total)]
        else:
            raise ValueError(f"Unsupported index file {index}")

    def detect_sample_loader(self):
        if self.index.endswith(".mdb"):
            return load_sample_lmdb
        elif self.index.endswith(".mat"):
            return load_sample_synthtext
        else:
            for loader in [load_sample_megane, load_sample_labelme]:
                try:
                    loader(self.samples[0])
                    return loader
                except Exception:
                    import traceback
                    traceback.print_exc()
                    continue

        raise ValueError("Invalid annotation, no compatible sample loader")

    def __getitem__(self, idx):
        image, annotation = self.load_sample(self.samples[idx])
        image.thumbnail((self.max_image_size, self.max_image_size))
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
