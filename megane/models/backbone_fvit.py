import math

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int):
        super().__init__()
        num_downscales = int(math.log2(patch_size))
        assert 2**num_downscales == patch_size
        self.input = nn.Conv2d(3, hidden_size, 1)
        self.embeds = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(hidden_size),
                    nn.ReLU(),
                )
                for _ in range(num_downscales)
            ]
        )

    def forward(self, images):
        patches = self.input(images)
        for embed in self.embeds:
            patches = embed(patches)
        return patches


class Embedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_size: int):
        super().__init__()
        assert image_size
        self.patch_embedding = PatchEmbedding(patch_size, hidden_size=hidden_size)
        with torch.no_grad():
            img = torch.rand(1, 3, image_size, image_size)
            img = self.patch_embedding(img)
        self.positional_encoding = nn.Parameter(
            torch.rand(1, hidden_size, *img.shape[-2:])
        )

    def forward(self, images):
        patches = self.patch_embedding(images)
        positions = self.positional_encoding.repeat([images.shape[0], 1, 1, 1])
        return positions + patches


class FactorAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.v_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.h_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)

    def forward(self, images):
        # Reshape to meet attention layer shape
        N, C, H, W = 0, 1, 2, 3
        images = images.permute(N, H, W, C)
        v_context, _ = torch.vmap(self.v_attention, in_dims=1, out_dims=1)(
            images, images, images
        )
        h_context, _ = torch.vmap(self.h_attention, in_dims=2, out_dims=2)(
            images, images, images
        )

        # Combine and permute back
        context = v_context + h_context
        N, H, W, C = 0, 1, 2, 3
        context = context.permute(N, C, H, W)
        return context


class Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.attention = FactorAttention(hidden_size, heads)
        self.norm_attention = nn.InstanceNorm2d(hidden_size)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=1),
            nn.ReLU(hidden_size),
            nn.Conv2d(hidden_size * 4, hidden_size, kernel_size=1),
        )
        self.norm_mlp = nn.InstanceNorm2d(hidden_size)

    def forward(self, images):
        ctx = self.norm_attention(images)
        images = self.attention(ctx) + images
        ctx = self.norm_mlp(images)
        images = self.mlp(ctx) + images
        return images


class Stage(nn.Module):
    def __init__(self, hidden_size, output_size, num_heads, num_blocks):
        super().__init__()
        blocks = [Block(hidden_size, num_heads) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv2d(hidden_size, output_size, 3, stride=2, padding=1)

    def forward(self, images):
        images = self.blocks(images)
        images = self.project(images)
        return images


class InvPatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size, output_size):
        super().__init__()
        num_upscales = int(math.log2(patch_size))
        assert 2**num_upscales == patch_size
        self.upscales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.InstanceNorm2d(hidden_size),
                    nn.ReLU(),
                )
                for _ in range(num_upscales)
            ]
        )
        self.output = nn.Conv2d(hidden_size, output_size, 1)

    def forward(self, patches):
        for upscale in self.upscales:
            patches = upscale(patches)
        return self.output(patches)


def output_sequence(image):
    # n c h w -> n c (h w)
    seq = image.flatten(-2)
    # n c (h w) -> n (h w) c
    seq = seq.transpose(-1, -2)
    return seq


def output_image(image):
    return image


class FViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.image_size % 256 == 0
        assert config.backbone.patch_size % 4 == 0
        assert config.backbone.output_format in ["seq", "image", "stages"]
        patch_size = config.backbone.patch_size
        num_blocks = config.backbone.num_blocks
        hidden_sizes = list(config.backbone.hidden_sizes)
        num_attention_heads = config.backbone.num_attention_heads

        num_stages = len(num_blocks)

        last_hidden = hidden_sizes[-1]
        # assert last_hidden % num_stages == 0
        # hidden_sizes[-1] = last_hidden // num_stages

        stages = []
        inv_embeds = []
        for stage_idx in range(num_stages):
            stage = Stage(
                hidden_sizes[stage_idx],
                hidden_sizes[stage_idx + 1],
                num_attention_heads[stage_idx],
                num_blocks[stage_idx],
            )
            stages.append(stage)

        self.patch_size = patch_size
        self.embedding = Embedding(
            config.image_size, patch_size, hidden_size=hidden_sizes[0]
        )
        self.stages = nn.ModuleList(stages)
        self.output_format = config.backbone.output_format
        if self.output_format == "seq":
            self.output = output_sequence
        elif self.output_format == "image":
            self.output = output_image

    def forward(self, images):
        patches = self.embedding(images)
        outputs = patches
        stage_outputs = []
        for stage in self.stages:
            outputs = stage(outputs)
            stage_outputs.append(outputs)
        if self.output_format == "stages":
            return stage_outputs
        else:
            return self.output(outputs)
