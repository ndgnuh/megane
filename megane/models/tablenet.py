# Code adapted from
# https://github.com/asagar60/TableNet-pytorch.git

import torch
from torchvision import models
from torch import nn
from torch.functional import F

from megane import registry


@registry.backbones.register("tablenet_vgg19")
class VGG19(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(VGG19, self).__init__()
        _vgg = models.vgg19(pretrained=pretrained).features
        self.vgg_pool3 = nn.Sequential()
        self.vgg_pool4 = nn.Sequential()
        self.vgg_pool5 = nn.Sequential()

        for x in range(19):
            self.vgg_pool3.add_module(str(x), _vgg[x])
        for x in range(19, 28):
            self.vgg_pool4.add_module(str(x), _vgg[x])
        for x in range(28, 37):
            self.vgg_pool5.add_module(str(x), _vgg[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        pool_3_out = self.vgg_pool3(x)  # torch.Size([1, 256, 128, 128])
        pool_4_out = self.vgg_pool4(pool_3_out)  # torch.Size([1, 512, 64, 64])
        pool_5_out = self.vgg_pool5(pool_4_out)  # torch.Size([1, 512, 32, 32])
        return (pool_3_out, pool_4_out, pool_5_out)


@registry.backbones.register("tablenet_densenet")
class DenseNet(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(DenseNet, self).__init__()
        denseNet = models.densenet121(pretrained=True).features
        self.densenet_out_1 = nn.Sequential()
        self.densenet_out_2 = nn.Sequential()
        self.densenet_out_3 = nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8, 10):
            self.densenet_out_2.add_module(str(x), denseNet[x])

        self.densenet_out_3.add_module(str(10), denseNet[10])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out_1 = self.densenet_out_1(x)  # torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1)  # torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2)  # torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3


class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=kernels[0], stride=strides[0]
        )
        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=kernels[1], stride=strides[1]
        )
        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2],
        )
        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3],
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  # [1, 256, 32, 32]
        out = self.upsample_1_table(x)  # [1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1)  # [1, 640, 64, 64]
        out = self.upsample_2_table(out)  # [1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1)  # [1, 512, 128, 128]
        out = self.upsample_3_table(out)  # [1, 3, 1024, 1024]
        return out


class ColumnDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ColumnDecoder, self).__init__()
        self.conv_8_column = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernels[0],
                stride=strides[0],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernels[0],
                stride=strides[0],
            ),
        )
        self.upsample_1_column = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=kernels[1], stride=strides[1]
        )
        self.upsample_2_column = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2],
        )
        self.upsample_3_column = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3],
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_8_column(x)  # [1, 256, 32, 32]
        out = self.upsample_1_column(x)  # [1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1)  # [1, 640, 64, 64]
        out = self.upsample_2_column(out)  # [1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1)  # [1, 512, 128, 128]
        out = self.upsample_3_column(out)  # [1, 3, 1024, 1024]
        return out


@registry.heads.register("tablenet_decoder")
class TableNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernels = [(1, 1), (2, 2), (2, 2), (8, 8)]
        self.strides = [(1, 1), (2, 2), (2, 2), (8, 8)]
        self.in_channels = 512
        self.pool_channels = [512, 256]

        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.strides = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=256, kernel_size=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )
        self.table_decoder = TableDecoder(
            self.pool_channels, self.kernels, self.strides
        )
        self.column_decoder = ColumnDecoder(
            self.pool_channels, self.kernels, self.strides
        )

    def forward(self, x, *a, **kwargs):
        pool_3_out, pool_4_out, pool_5_out = x
        conv_out = self.conv6(pool_5_out)  # [1, 256, 32, 32]
        table_out = self.table_decoder(
            conv_out, pool_3_out, pool_4_out
        )  # torch.Size([1, 1, 1024, 1024])
        column_out = self.column_decoder(
            conv_out, pool_3_out, pool_4_out
        )  # torch.Size([1, 1, 1024, 1024])
        return table_out, column_out

    def compute_loss(self, outputs, targets):
        table, column = outputs
        loss = 0
        loss += F.binary_cross_entropy_with_logits(table.squeeze(1), targets[:, 0])
        loss += F.binary_cross_entropy_with_logits(column.squeeze(1), targets[:, 1])
        return loss

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth=False):
        if not ground_truth:
            outputs = torch.cat(outputs, dim=1)
            outputs = torch.sigmoid(outputs)
        # else:
        #     ic(outputs.shape)

        if outputs.ndim == 4:
            outputs = torch.cat(list(outputs), dim=-2)
        outputs = torch.cat(list(outputs), dim=-1)
        outputs = outputs.unsqueeze(0)
        logger.add_image(tag, outputs, step)
