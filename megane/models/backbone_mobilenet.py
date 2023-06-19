from typing import List

import torch
from torch import nn

from megane.models.utils import AFAttention, Chain


class AFTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ctx_length: int,
        dropout: float = 0.0,
        act: callable = nn.ReLU,
    ):
        super().__init__()
        self.attention = AFAttention(hidden_size, ctx_length)
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Forward attention
        residual = x
        x = self.norm_attention(x)
        x = self.attention(x)
        x = x + residual

        # Forward mlp
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + residual

        return x


class MV2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand=6,
    ):
        super().__init__()
        mid_channels = in_channels * expand
        self.features = Chain(
            pconv1=nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            pn1=nn.BatchNorm2d(mid_channels),
            pa1=nn.ReLU6(),
            dconv=nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                groups=mid_channels,
                bias=False,
                padding=1,
            ),
            dn=nn.BatchNorm2d(mid_channels),
            da=nn.ReLU6(),
            pconv2=nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            pn2=nn.BatchNorm2d(out_channels),
        )
        self.use_residual = (in_channels == out_channels) and stride == 1

    def forward(self, imgs):
        outputs = self.features(imgs)
        if self.use_residual:
            outputs = imgs + outputs
        return outputs


class MViTBlock(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, ft_map_size, stride=1):
        super().__init__()
        # Local representation block
        self.local_representation = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.Conv2d(hidden_size, hidden_size, 1),
        )

        # Transformer block
        layers = []
        ctx_length = ft_map_size**2
        for i in range(num_layers):
            layer = AFTransformerLayer(hidden_size, ctx_length)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        # Fusion block
        self.fusion_conv1 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.fusion_conv2 = nn.Conv2d(
            hidden_size * 2,
            output_size,
            kernel_size=3,
            padding=1,
            stride=stride,
        )

    def unfold(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1)
        x = x.transpose(1, -1)
        return x, h, w

    def fold(self, x, h, w):
        n, l, c = x.shape
        assert h * w == l
        x = x.transpose(-1, 1)
        x = x.reshape(n, c, h, w)
        return x

    def forward(self, x):
        # x: N C H W
        residual = x
        # Unfold

        # local repr
        x = self.local_representation(x)

        # transformer
        x, h, w = self.unfold(x)
        x = self.layers(x)
        x = self.fold(x, h, w)

        # fusion
        x = self.fusion_conv1(x)
        x = torch.cat([x, residual], dim=1)
        x = self.fusion_conv2(x)
        return x


# class MobileViT(nn.Module):
#     def __init__(
#         self,
#         image_size: int,
#         hidden_sizes: List[int],
#     ):
#         super().__init__()
#         self.stem = nn.Conv2d(
#             in_channels=3,
#             out_channels=16,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         )

#         # Short word
#         c = hidden_sizes
#         B1 = MV2Block
#         B2 = MViTBlock


#         configs = [
#             # stage 1
#             (B1, 16, c[0], 1),
#             # stage 2
#             (B1, c[0], c[1], 2),
#             (B1, c[1], c[1], 1),
#             (B1, c[1], c[1], 1),
#             # stage 3
#             (B1, c[1], c[2], 2),
#             (B2, c[1],
#         ]


class AFViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        hidden_sizes: List[int],
        num_layers: List[int],
        project_size: int,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, hidden_sizes[0], 3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_sizes[0]),
            nn.ReLU(),
        )

        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()

        num_stages = len(hidden_sizes) - 1
        ft_map_size = image_size // 2

        for i in range(num_stages):
            stage = MViTBlock(
                hidden_size=hidden_sizes[i],
                output_size=hidden_sizes[i + 1],
                num_layers=num_layers[i],
                ft_map_size=ft_map_size,
                stride=2,
            )
            ft_map_size = ft_map_size // 2
            self.stages.append(stage)
            self.projections.append(
                nn.Conv2d(
                    hidden_sizes[i + 1],
                    project_size,
                    kernel_size=1,
                )
            )

    def forward(self, x):
        x = self.patch_embedding(x)
        outputs = []
        for i, stage in enumerate(self.stages):
            project = self.projections[i]
            x = stage(x)
            outputs.append(project(x))
        return outputs


def afvit_t(image_size: int, project_size: int):
    return AFViT(
        image_size=image_size,
        num_layers=[3, 3, 3, 3],
        hidden_sizes=[32, 64, 96, 128, 192],
        project_size=project_size,
    )
