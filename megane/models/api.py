from typing import TypeVar
from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.utils.data import default_collate
from ..data import Sample

T = TypeVar("T")


class ModelAPI(nn.Module, ABC):
    @abstractmethod
    def encode_sample(self, sample: Sample) -> T:
        ...

    @abstractmethod
    def decode_sample(self,
                      inputs: Tensor, outputs: Tensor,
                      ground_truth: bool = False) -> Sample:
        ...

    @abstractmethod
    def compute_loss(self, outputs, targets) -> Tensor:
        ...

    @abstractmethod
    def visualize_outputs(self, outputs, grouth_truth: bool = False) -> Tensor:
        ...

    def collate_fn(self, samples):
        return default_collate(samples)
