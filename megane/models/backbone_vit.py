from torch import nn
from megane.registry import backbones


class SeparableSelfAttention(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(hidden_dims, 1), nn.Softmax(dim=1))
        self.key = nn.Linear(hidden_dims, hidden_dims)
        self.value = nn.Linear(hidden_dims, hidden_dims)
        self.project = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x, extra_query=0):
        q = self.query(x) + extra_query
        k = self.key(x)
        v = self.value(x)

        ctx = q * k
        ctx = ctx.sum(dim=1, keepdim=True)
        ctx = self.project(ctx * v)
        return ctx


def Stem(hidden_size: int):
    return nn.Sequential(
        nn.Conv2d(3, hidden_size, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.Conv2d(hidden_size, hidden_size, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
    )


def MLP(hidden_size: int, dropout: float = 0.0):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size * 4, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_size * 4, hidden_size, bias=False),
        nn.Dropout(dropout),
    )


class Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.attn = SeparableSelfAttention(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x):
        # Attention
        ctx = self.norm_attn(x)
        ctx = self.attn(ctx)
        x = ctx + x

        # MLP
        ctx = self.norm_mlp(x)
        ctx = self.mlp(x)
        x = ctx + x
        return x


class Stage(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_layers: int):
        super().__init__()
        blocks = [Block(input_size) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        n, c, h, w = x.shape

        # unfold
        # x: n c h w -> n (h w) c
        x = x.reshape(n, c, h * w)
        x = x.permute(0, 2, 1)

        # transformer forward
        x = self.blocks(x)

        # fold
        # x: n (h w) c -> n c h w
        x = x.permute(0, 2, 1)
        x = x.reshape(n, c, h, w)
        return x


class MobileViT(nn.Sequential):
    def __init__(self, hidden_sizes, num_layers):
        super().__init__()

        # Stem layer w / 4 h / 4 layer
        self.stem = Stem(hidden_sizes[0])

        count = 0
        for item in zip(hidden_sizes, hidden_sizes[1:], num_layers):
            # Stage
            input_size, output_size, num_layer = item
            stage = Stage(input_size, output_size, num_layer)
            stage_name = f"stage_{count}"
            setattr(self, stage_name, stage)

            # Down sample projection layer
            # feature map size: w / 2^(count + 1) / 4
            layer = nn.Sequential(
                nn.Conv2d(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(output_size),
                nn.ReLU(),
            )
            setattr(self, f"project_{count}", layer)

            # Increate count
            count = count + 1


@backbones.register()
def mobilevit11():
    hidden_sizes = [24, 48, 64, 96, 128]
    num_layers = [2, 2, 2, 2]
    return MobileViT(hidden_sizes, num_layers)


@backbones.register()
def mobilevit18():
    hidden_sizes = [32, 64, 128, 192, 192]
    num_layers = [4, 4, 4, 4]
    return MobileViT(hidden_sizes, num_layers)


@backbones.register()
def mobilevit50():
    hidden_sizes = [64, 128, 256, 512, 512]
    num_layers = [3, 4, 6, 3]
    return MobileViT(hidden_sizes, num_layers)
