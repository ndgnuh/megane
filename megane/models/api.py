from abc import ABC, abstractmethod
from typing import TypeVar

from torch import Tensor, nn
from torch.utils.data import default_collate

from ..data import Sample

T = TypeVar("T")


class ModelAPI(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, outputs, targets) -> Tensor:
        ...

    @abstractmethod
    def visualize_outputs(
        self,
        outputs,
        logger,
        tag: str,
        step: int,
        grouth_truth: bool = False,
    ) -> Tensor:
        ...
