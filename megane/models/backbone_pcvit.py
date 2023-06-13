import math
import torch
from torch import nn

from ..configs import PCViTConfig


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int):
        super().__init__()
        num_downscales = int(math.log2(patch_size))
        assert 2**num_downscales == patch_size
        self.input = nn.Conv2d(3, hidden_size, 1)
        self.embeds = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, 2, stride=2),
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


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
            nn.ReLU(),
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, images):
        ctx = self.norm_attention(images)
        images = self.attention(ctx, ctx, ctx)[0] + images
        ctx = self.norm_mlp(images)
        images = self.mlp(ctx) + images
        return images


class IOAdapter(nn.Module):
    def __init__(
        self, input_size, output_size, output_length, num_attention_heads: int
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.rand(1, output_length, output_size))
        self.attn = nn.MultiheadAttention(
            output_size,
            num_attention_heads,
            kdim=input_size,
            vdim=input_size,
            batch_first=True,
        )

    def forward(self, x):
        L = self.latents.repeat([x.size(0), 1, 1])
        ctx, _ = self.attn(L, x, x)
        return ctx


class PCViTBackbone(nn.Module):
    def __init__(self, config: PCViTConfig):
        super().__init__()
        bc = config.backbone  # backbone config
        image_size = config.image_size
        self.final_size = image_size // bc.final_div_factor
        self.aux_size = self.final_size // 4

        layers = []
        self.embed = Embedding(image_size, bc.patch_size, bc.hidden_size)
        self.input_adapter = IOAdapter(
            bc.hidden_size, bc.latent_size, bc.num_latents, bc.num_attention_heads
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(bc.latent_size, bc.num_attention_heads)
                for _ in range(bc.num_layers)
            ]
        )
        self.output_adapter = IOAdapter(
            bc.latent_size,
            bc.output_size,
            self.aux_size**2,
            bc.num_attention_heads,
        )
        self.upscale = nn.Sequential(
            UpscaleConv(bc.output_size),
            UpscaleConv(bc.output_size),
        )

    def image2seq(self, x):
        b, c, h, w = 0, 1, 2, 3
        x = x.permute(b, h, w, c)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x

    def seq2image(self, x):
        x = x.reshape(x.size(0), self.aux_size, self.aux_size, x.size(2))
        b, h, w, c = 0, 1, 2, 3
        x = x.permute(b, c, h, w)
        return x

    def forward(self, x):
        x = self.embed(x)
        b, c, h, w = x.shape
        x = self.image2seq(x)
        x = self.input_adapter(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_adapter(x)
        x = self.seq2image(x)
        x = self.upscale(x)
        return x


class UpscaleConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, 2, 2, groups=channels)
        self.norm_up = nn.InstanceNorm2d(channels)
        self.pj = nn.Conv2d(channels, channels, 1)
        self.norm_pj = nn.InstanceNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, images):
        images = self.up(images)
        images = self.norm_pj(images)
        images = self.pj(images)
        images = self.act(images)
        return images
