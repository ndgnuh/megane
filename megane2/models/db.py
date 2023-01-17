from torch import nn
import torch


class DBHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        num_upscales: int = 2,
        activation: str = 'ReLU'
    ):
        super().__init__()
        self.num_upscales = num_upscales
        self.num_classes = num_classes
        self.input_size = input_size
        self.activation = activation
        assert hasattr(nn, activation), f"Invalid activation {activation}"

        self.prob_head = self.make_branch()
        self.thres_head = self.make_branch()

    def forward(self, features, returns_threshold=False):
        if self.training or returns_threshold:
            thres = self.thres_head(features)
        else:
            thres = None
        prob = self.prob_head(features)
        return prob, thres

    def make_branch(self):
        input_size = self.input_size
        head_size_d4 = self.input_size // 4

        return nn.Sequential(
            nn.Conv2d(input_size, head_size_d4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_size_d4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_size_d4, head_size_d4,
                               2, stride=2, bias=False),
            nn.BatchNorm2d(head_size_d4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_size_d4, self.num_classes, 2, stride=2),
        )
