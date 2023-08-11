from typing import List

import torch
from lightning import LightningModule
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss


class ConvBR(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.norm = nn.BatchNorm2d(self.conv.out_channels)
        self.relu = nn.ReLU()

    def __repr__(self):
        return repr(self.conv).replace("Conv2d", "ConvBatchNormReLU")


def VGGStage(in_channels: int, out_channels: int, num_layers: int):
    layers = [ConvBR(in_channels, out_channels, 3, padding=1)]
    layers = layers + [
        ConvBR(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        for _ in range(num_layers - 1)
    ]
    layers = layers + [nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        for i, (i, o, k) in enumerate(config):
            stage = VGGStage(i, o, k)
            self.stages.append(stage)

    def forward(self, x):
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
        return outputs


def vgg19bn():
    config = [
        (3, 64, 2),
        (64, 128, 2),
        (128, 256, 4),
        (256, 512, 4),
        (512, 512, 4),
    ]
    return VGG(config)


def vgg19bn_thin():
    config = [
        (3, 16, 2),
        (16, 32, 2),
        (32, 64, 4),
        (64, 92, 4),
        (92, 92, 4),
    ]
    return VGG(config)


class FCOSHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, stride: int):
        super().__init__()
        self.localize = nn.Sequential(
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
        )
        self.regression = nn.Conv2d(hidden_size, 4 * num_classes, 3, padding=1)
        self.centerness = nn.Conv2d(hidden_size, num_classes, 3, padding=1)
        self.classification = nn.Sequential(
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            ConvBR(hidden_size, hidden_size, 3, padding=1),
            nn.Conv2d(hidden_size, num_classes, 3, padding=1),
        )
        self.stride = stride
        self.num_classes = num_classes

    def lazy_grid(self, ft_map):
        if not hasattr(self, "grid"):
            _, _, H, W = ft_map.shape
            x = torch.arange(H, device=ft_map.device)[None, :]
            y = torch.arange(W, device=ft_map.device)[:, None]
            y, x = torch.broadcast_tensors(y, x)
            grid = torch.stack([y, x], dim=0)
            self.register_buffer("grid", grid)

        y, x = self.grid.unbind(0)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return y, x

    def forward(self, ft_map: Tensor):
        loc = self.localize(ft_map)
        ctn = self.centerness(loc)
        reg = self.regression(loc)
        cls = self.classification(ft_map)

        # Original paper
        reg = torch.relu(reg)
        reg = reg.reshape(reg.size(0), self.num_classes, 4, reg.size(2), reg.size(3))
        # Instead, add the position so we regress the changes
        # Note: Negative position got boosted too...
        # l, t, r, b = reg.unbind(dim=1)
        # y, x = self.lazy_grid(ft_map)
        # l = l + x
        # r = r + x
        # t = t + y
        # b = b + y
        # # ic(l.shape)
        # reg = torch.stack([l, t, r, b], dim=1)
        return reg, ctn, cls


class FeaturePyramid(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        num_extra_layers: int = 2,
    ):
        super().__init__()
        self.fpn_projects = nn.ModuleList(
            [ConvBR(h_size, output_size, 1) for h_size in hidden_sizes]
        )
        self.fpn_extras = nn.ModuleList(
            [
                ConvBR(output_size, output_size, 3, padding=1, stride=2)
                for _ in range(num_extra_layers)
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, ft_maps):
        n = len(ft_maps)

        # P3, P4, P5
        fpn_maps = []
        for i in range(n):
            layer = self.fpn_projects[n - i - 1]
            ft = ft_maps[n - i - 1]
            if i == 0:
                fpn_ft = layer(ft)
                fpn_maps.append(fpn_ft)
            else:
                prev = self.upsample(fpn_maps[i - 1])
                fpn_ft = layer(ft) + prev
                fpn_maps.append(fpn_ft)

        # P6, P7
        for layer in self.fpn_extras:
            ft = fpn_maps[0]
            new_ft = layer(ft)
            fpn_maps.insert(0, new_ft)

        return fpn_maps


class FCOS(LightningModule):
    def __init__(
        self,
        head_size: int,
        num_classes: int,
        backbone: str = "vgg19bn_thin",
        num_heads: int = 5,
        base_stride: int = 8,
    ):
        super().__init__()
        backbone = eval(backbone)
        # Backbone
        self.backbone = backbone()
        with torch.no_grad():
            image = torch.rand(1, 3, 800, 1024)
            hidden_sizes = [x.shape[1] for x in self.backbone(image)][-3:]

        # FPN
        num_extra_layers = num_heads - len(hidden_sizes)
        self.fpn = FeaturePyramid(
            hidden_sizes=hidden_sizes,
            output_size=head_size,
            num_extra_layers=num_extra_layers,
        )

        # Prediction head
        strides = [base_stride * 2**i for i in range(num_heads)]
        self.heads = nn.ModuleList(
            [FCOSHead(head_size, num_classes, stride) for stride in strides],
        )
        self.num_heads = num_heads
        for layer in self.heads.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if hasattr(layer, "bias"):
                    nn.init.constant_(layer.bias, 0)
        self.num_classes = num_classes

    def forward(self, x):
        # Take c3, c4 and c5
        ft_maps = self.backbone(x)[-3:]

        # Produce p3 -> p7
        ft_maps = self.fpn(ft_maps)

        # Predict at multi level
        centerness_maps = []
        regression_maps = []
        classification_maps = []
        for i in range(self.num_heads):
            layer = self.heads[i]
            ft = ft_maps[i]
            reg, ctn, cls = layer(ft)
            regression_maps.insert(0, reg)
            centerness_maps.insert(0, ctn)
            classification_maps.insert(0, cls)

        # Done, tada!
        return regression_maps, centerness_maps, classification_maps

    def training_step(self, batch):
        (
            image,
            gt_regression_maps,
            gt_centerness_maps,
            gt_classification_maps,
            training_maps,
        ) = batch
        regression_maps, centerness_maps, classification_maps = self(image)
        losses = []
        cls_losses = []
        rgs_losses = []
        ctn_losses = []
        batch_size = image.shape[0]
        for pr, gt, pos in zip(
            classification_maps, gt_classification_maps, training_maps
        ):
            # loss = sigmoid_focal_loss(pr, gt).mean()

            neg = ~pos
            loss = 0
            if torch.count_nonzero(pos) > 0:
                loss += F.binary_cross_entropy_with_logits(pr[pos], gt[pos])
            if torch.count_nonzero(neg) > 0:
                loss += F.binary_cross_entropy_with_logits(pr[neg], gt[neg])
            cls_losses.append(loss)

        for pr, gt, pos in zip(
            regression_maps,
            gt_regression_maps,
            training_maps,
        ):
            # B4HW -> BHW4 -> BL4
            # pos = pos.flatten()
            # if pos.count_nonzero() == 0:
            #     # loss = F.l1_loss(pr, gt)
            #     continue
            # else:
            #     pr = pr.permute(0, 2, 3, 1).flatten(0, 2)
            #     gt = gt.permute(0, 2, 3, 1).flatten(0, 2)

            # Merge NC4HW -> (NC) 4 H W
            loss = 0
            for i in range(self.num_classes):
                pos_c = pos[:, i]
                pr_c = pr[:, i]
                gt_c = gt[:, i]
                if torch.count_nonzero(pos_c) == 0:
                    loss += F.l1_loss(pr_c, gt_c) * 0.25
                else:
                    loss += iou_loss(pr_c, gt_c)[pos_c].mean()

            if loss > 0:
                rgs_losses.append(loss)

        for pr, gt, pos in zip(centerness_maps, gt_centerness_maps, training_maps):
            # loss = F.binary_cross_entropy_with_logits(pr, gt)

            neg = ~pos
            loss = 0
            if torch.count_nonzero(pos) > 0:
                loss += F.binary_cross_entropy_with_logits(pr[pos], gt[pos])
            if torch.count_nonzero(neg) > 0:
                loss += F.binary_cross_entropy_with_logits(pr[neg], gt[neg])
            ctn_losses.append(loss)

        # losses = [
        #     loss for loss in losses if not torch.isnan(loss) and not torch.isinf(loss)
        # ]
        cls_loss = sum(cls_losses) / len(cls_losses)
        ctn_loss = sum(ctn_losses) / len(ctn_losses)
        rgs_loss = sum(rgs_losses) / len(rgs_losses)
        loss = (ctn_loss + cls_loss + rgs_loss) / 3

        loss_item = loss.item()
        self.log("loss", loss_item, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=1e-2,
            weight_decay=1e-5,
            momentum=0.9,
        )
        my_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[500, 5000, 10000], gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": my_lr_scheduler,
            "interval": "step",
        }

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), "model.pt")

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        if (1 + batch_idx) % 100 != 0:
            return
        print("Model saved to model.pt")
        torch.save(self.state_dict(), "model.pt")


def iou_loss(pr, gt):
    # Unpack
    pr_l, pr_t, pr_r, pr_b = pr.unbind(1)
    gt_l, gt_t, gt_r, gt_b = gt.unbind(1)

    # Area
    pr_area = (pr_r + pr_l) * (pr_b + pr_t)
    gt_area = (gt_r + gt_l) * (gt_b + gt_t)

    # IOU
    w_inter = torch.min(pr_l, gt_l) + torch.min(pr_r, gt_r)
    h_inter = torch.min(pr_b, gt_b) + torch.min(pr_t, gt_t)
    a_inter = w_inter * h_inter
    a_union = gt_area + pr_area - a_inter
    iou = (1 + a_inter) / (1 + a_union)

    return -torch.log(iou)
