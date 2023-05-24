from dataclasses import dataclass
from typing import *

from .configs import TrainConfig, ModelConfig

@dataclass
class BatchEncoding:
    batch_size: int
    data: Dict

    def __getitem__(self, i):
        if isinstance(i, int):
            return {k: v[i]  for k, v in self.data.items()}
        else:
            return self.data[i]
