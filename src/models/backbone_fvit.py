from typing import List
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size)
        self.norm = nn.InstanceNorm2d(hidden_size)
        self.act = nn.ReLU()

    def forward(self, images):
        patches = self.conv(images)
        patches = self.norm(patches)
        patches = self.act(patches)
        return patches


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
            nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=1, groups=hidden_size),
            nn.ReLU(hidden_size),
            nn.Conv2d(hidden_size * 4, hidden_size, kernel_size=1, groups=hidden_size),
            nn.Dropout2d(0.1),
        )
        self.norm_mlp = nn.InstanceNorm2d(hidden_size)

    def forward(self, images):
        images = self.norm_attention(images)
        images = self.attention(images) + images
        images = self.norm_mlp(images)
        images = self.mlp(images) + images
        return images


class Stage(nn.Module):
    def __init__(self, hidden_size, output_size, num_heads, num_blocks):
        super().__init__()
        blocks = [Block(hidden_size, num_heads) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.downscale = nn.Conv2d(hidden_size, output_size, 3, stride=2, padding=1)

    def forward(self, images):
        images = self.blocks(images)
        images = self.downscale(images)
        return images


class InvPatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size, output_size):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_size, hidden_size, patch_size, patch_size, groups=hidden_size
            ),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, 1),
            nn.InstanceNorm2d(output_size),
            nn.ReLU(),
        )

    def forward(self, patches):
        return self.tconv(patches)


class FViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.image_size % 256 == 0
        assert config.backbone.patch_size % 4 == 0
        patch_size = config.backbone.patch_size
        num_blocks = config.backbone.num_blocks
        hidden_sizes = config.backbone.hidden_sizes
        num_attention_heads = config.backbone.num_attention_heads
        num_stages = len(num_blocks)
        assert hidden_sizes[-1] % 3 == 0

        stages = []
        inv_embeds = []
        for stage_idx in range(num_stages):
            stage = Stage(
                hidden_sizes[stage_idx],
                hidden_sizes[stage_idx + 1],
                num_attention_heads[stage_idx],
                num_blocks[stage_idx],
            )
            inv_embed = InvPatchEmbedding(
                patch_size // 2 * (2**stage_idx),
                hidden_size=hidden_sizes[stage_idx + 1],
                output_size=hidden_sizes[-1] // 3,
            )
            stages.append(stage)
            inv_embeds.append(inv_embed)

        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(patch_size, hidden_size=hidden_sizes[0])
        self.stages = nn.ModuleList(stages)
        self.inv_patch_embeddings = nn.ModuleList(inv_embeds)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(hidden_sizes[-1], hidden_sizes[-1], 2, 2),
            nn.InstanceNorm2d(hidden_sizes[-1]),
            nn.ReLU(),
            nn.Conv2d(hidden_sizes[-1], hidden_sizes[-1], 3, padding=1),
        )

    def forward(self, images):
        patches = self.patch_embedding(images)
        stage_outputs = []
        stage_output = patches
        for stage, inv_embed in zip(self.stages, self.inv_patch_embeddings):
            stage_output = stage(stage_output)
            scale_embed = inv_embed(stage_output)
            stage_outputs.append(scale_embed)

        outputs = torch.cat(stage_outputs, dim=1)
        outputs = self.output(outputs)
        return outputs
