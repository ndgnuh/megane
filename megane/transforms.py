from . import ops, configs
from dataclasses import dataclass
from PIL import Image
from typing import List, Callable
import numpy as np


@dataclass
class Compose:
    transforms: List[Callable]

    def __call__(self, image: Image.Image, annotation):
        for transform in self.transforms:
            image, annotation = transform(image, annotation)
        return image, annotation


@dataclass
class Resize:
    width: int
    height: int

    def __call__(self, image: Image.Image, annotation):
        return image.resize((self.width, self.height)), annotation


@dataclass
class DBPreprocess:
    image_width: int
    image_height: int
    shrink_ratio: float = 0.4
    min_box_size: int = 10

    @classmethod
    def from_config(cls, config):
        keys = ["image_width", "image_height", "shrink_ratio", "min_box_size"]
        return configs.init_from_config(cls, config, keys)

    def __call__(self, image: Image.Image, annotation):
        import torch
        from torchvision.transforms.functional import to_tensor

        spade_loss_mask= ops.polygon_to_mask_segment(image,annotation,self.image_width, self.image_height)
        # print("spade_loss_mask: ",type(spade_loss_mask))
        spade_loss_mask=torch.tensor(spade_loss_mask).type(torch.bool)

        image = image.resize((self.image_width, self.image_height))

        polygons = np.array(annotation['shapes'][0]['points'])
        labels = np.array(annotation['shapes'][0]['label'])

        # Proba map/mask, threshold map/mask
        targets = []
        width, height = image.size
        label_set = np.unique(labels)
        label_set.sort()
        for label in label_set:
            polygons_ = polygons[labels == label]
            targets_ = ops.build_db_target(
                polygons_,
                image_width=width,
                image_height=height,
                shrink_ratio=self.shrink_ratio,
                min_box_size=self.min_box_size
            )
            targets.append(targets_)

        # Stack label channel
        targets = [
            np.stack([target[i] for target in targets])
            for i in range(4)
        ]
        
        # To tensor
        image = to_tensor(image)
        proba_maps = torch.tensor(targets[0]).type(torch.bool)
        threshold_maps = torch.tensor(targets[2]) / 255
        proba_masks = torch.tensor(targets[1]).type(torch.bool)
        theshold_masks = torch.tensor(targets[3]).type(torch.bool)
        return image, (proba_maps, proba_masks, threshold_maps, theshold_masks,spade_loss_mask)


@ dataclass
class DBPostprocess:
    expand_ratio: float = 10
    min_box_size: int = 10
    min_score: float = 0.6
    min_threshold: float = 0.7

    def __call__(self, proba_maps: np.ndarray):
        polygons, labels, scores, angles = [], [], [], []
        for label, proba_map in enumerate(proba_maps):
            polygons_, scores_, angles_ = ops.mask_to_polygons(
                proba_map,
                expand_ratio=self.expand_ratio,
                min_box_size=self.min_box_size,
                min_score=self.min_score,
                min_threshold=self.min_threshold
            )
            polygons.extend(polygons_)
            angles.extend(angles_)
            scores.extend(scores_)
            labels.extend([label] * len(scores_))
        return polygons, angles, labels, scores

    @ classmethod
    def from_config(cls, config):
        return cls(
            min_threshold=config.get('min_threshold', 0.7),
            ** {k: config[k] for k in [
                'expand_ratio',
                'min_box_size',
                'min_score'
            ]}
        )

# class DBPostprocess:
#     expand_ratio: float = 10
#     min_box_size: int = 10
#     min_score: float = 0.6
#     min_threshold: float = 0.7

#     def __call__(self, proba_maps: np.ndarray):
#         polygons, labels, scores, angles = [], [], [], []
#         for label, proba_map in enumerate(proba_maps):
#             polygons_, scores_, angles_ = ops.mask_to_polygons(
#                 proba_map,
#                 expand_ratio=self.expand_ratio,
#                 min_box_size=self.min_box_size,
#                 min_score=self.min_score,
#                 min_threshold=self.min_threshold
#             )
#             polygons.extend(polygons_)
#             angles.extend(angles_)
#             scores.extend(scores_)
#             labels.extend([label] * len(scores_))

#         return polygons, angles, labels, scores

#     @ classmethod
#     def from_config(cls, config):
#         return cls(
#             min_threshold=config.get('min_threshold', 0.7),
#             ** {k: config[k] for k in [
#                 'expand_ratio',
#                 'min_box_size',
#                 'min_score'
#             ]}
#         )
