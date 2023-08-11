from typing import List

import icecream
import torch
from lightning import LightningModule, Trainer
from src.encode import encode_fcos
from src.labelme import LabelMeDataset
from src.net import FCOS
from torch.utils.data import DataLoader

image_size = (800, 999999999999)
if False:
    batch_size = 1
    shuffle = False
    limit_train_batches = 1
    class2str, index_file = ["dog", "cat"], "./index.txt"
else:
    batch_size = 4
    shuffle = True
    limit_train_batches = 1.0
    class2str, index_file = ["table", "column"], "./data/positive.txt"
num_classes = len(class2str)
icecream.install()


def transform(image, boxes, classes):
    return encode_fcos(image, boxes, classes, W=1024, H=1024, C=num_classes)


dataset = LabelMeDataset(index_file, class2str, transform=transform)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=batch_size - 1,
    shuffle=shuffle,
)
batch = next(iter(train_loader))
trainer = Trainer(
    max_epochs=10000,
    accelerator="auto",
    limit_train_batches=limit_train_batches,
)
model = FCOS(128, 2)
model.load_state_dict(torch.load("model.pt"), strict=False)
trainer.fit(model, train_loader)
