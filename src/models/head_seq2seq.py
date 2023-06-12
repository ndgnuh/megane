from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .api import ModelAPI
from ..data import Sample
from ..utils import prepare_input, polygon2xyxy, xyxy2polygon


class BatchList(list):
    def __getitem__(self, item):
        return [t[item] for t in self]


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        outputs, last_hidden = self.rnn(x)
        last_hidden = torch.cat(last_hidden.chunk(2, dim=0), dim=-1).squeeze(0)
        last_hidden = self.fc(last_hidden)
        return outputs, last_hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        pass


class Decode(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()


class Seq2seq(ModelAPI):
    def __init__(self, config):
        super().__init__()
        head_size = config.head.head_size
        num_max_objects = config.head.num_max_objects

        # Meta
        self.num_max_objects = num_max_objects
        self.image_size = config.image_size

        # Encoding
        self.encode = Encoder(config.hidden_size, head_size)
        self.decode = nn.GRU(head_size, head_size)

        # detections
        self.localize = nn.Linear(head_size, 4, bias=False)

        # classification
        self.classify = nn.Linear(
            head_size, config.head.num_classes + 1, bias=False)

        # auxilary task
        self.count = nn.Sequential(
            nn.LayerNorm(head_size),
            nn.Linear(head_size, num_max_objects)
        )
        raise RuntimeError("This idea DOES NOT WORK, use something else")

    def forward(self, features, targets=None):
        # Feature map
        outputs, hidden = self.encode(features)

        # Object counting
        count = self.count(hidden)
        if targets is None:
            object_count = count.argmax(dim=-1).max().item()
        else:
            _, classes, _ = targets
            object_count = classes.shape[-1]

        boxes, class_logits = [], []
        inp, hidden = hidden, None
        for _ in range(object_count):
            inp, hidden = self.decode(inp, hidden)
            box = self.localize(inp)
            logits = self.classify(inp)
            boxes.append(box)
            class_logits.append(logits)

        boxes = torch.stack(boxes, dim=0).transpose(0, 1)
        class_logits = torch.stack(class_logits, dim=0).transpose(0, 1)
        return BatchList([boxes, class_logits, count])

    def encode_sample(self, sample: Sample):
        inputs = prepare_input(sample.image, self.image_size, self.image_size)
        inputs = torch.FloatTensor(inputs)
        boxes = torch.FloatTensor([polygon2xyxy(box) for box in sample.boxes])
        classes = torch.LongTensor(sample.classes) + 1
        num_targets = len(sample.classes)
        assert num_targets <= self.num_max_objects
        return inputs, boxes, classes, num_targets

    def collate_fn(self, samples):
        images = []
        classes = []
        boxes = []
        num_targets = []
        max_length: int = max(num_targets for _, _, _, num_targets in samples)
        for (image, boxes_, classes_, num_targets_) in samples:
            images.append(image)
            num_targets.append(num_targets_)

            boxes_ = F.pad(boxes_, [0, 0, 0, max_length - num_targets_])
            boxes.append(boxes_)
            classes_ = F.pad(classes_, [0, max_length - num_targets_])
            classes.append(classes_)

        images = torch.stack(images, dim=0)
        num_targets = torch.LongTensor(num_targets)
        classes = torch.stack(classes, dim=0)
        boxes = torch.stack(boxes, dim=0)
        return images, BatchList([boxes, classes, num_targets])

    def compute_loss(self, outputs, targets):
        pr_locs, pr_cls, pr_counts = outputs
        gt_locs, gt_cls, gt_counts = targets

        # Localization loss
        loc_loss = F.mse_loss(pr_locs, gt_locs)

        # Classification loss
        cls_loss = F.cross_entropy(pr_cls.transpose(-1, 1), gt_cls)

        # Counting loss
        count_loss = F.cross_entropy(pr_counts, gt_counts)
        loss = loc_loss + count_loss + cls_loss
        return loss

    def decode_sample(self, inputs, outputs, ground_truth=False):
        # Unpack
        try:
            locs, class_logits, _ = outputs
        except Exception:
            ic(outputs)

        # Decode classification
        if not ground_truth:
            scores, classes = torch.softmax(class_logits, dim=-1).max(dim=-1)
            classes = classes - 1
            mask = classes >= 0
            scores = scores[mask]
            classes = classes[mask]
        else:
            classes = class_logits
            scores = torch.ones_like(classes)

        # Convert to sample
        image = TF.to_pil_image(inputs)
        boxes = [xyxy2polygon(b) for b in locs.cpu().detach().numpy()]
        classes = classes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        return Sample(image=image,
                      boxes=boxes,
                      classes=classes,
                      scores=scores)

    def visualize_outputs(self, outputs, **k):
        return torch.ones(1, 10, 10)
