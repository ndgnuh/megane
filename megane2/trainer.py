import cv2
import torch
from megane2.loaders import megane_dataloader
from megane2 import transforms, losses, models, scores
from pytorch_lightning.lite import LightningLite
from torch import optim
from tqdm import tqdm
from . import visualize
import random


def cycle(dataloader, total_steps):
    step = 0
    while True:
        for batch in dataloader:
            step = step + 1
            yield step, batch
            if step == total_steps:
                return


class Trainer(LightningLite):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        train_data: str,
        val_data: str,
        total_steps: int = 10_000,
        validate_every: int = 100,
        batch_size: int = 4,
        num_workers: int = 1,
    ):
        super().__init__(accelerator="auto")
        self.total_steps = total_steps
        self.validate_every = validate_every

        # Model
        self.criterion = losses.DBLoss()
        self.model = models.DBNet("mobilenet_v3_large", 96)
        self.post_processor = transforms.DBPostprocess(expand_ratio=1.5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        # datasets
        transform = transforms.Compose([
            transforms.Resize(image_width, image_height),
            transforms.DBPreprocess(),
        ])

        self.train_loader = megane_dataloader(
            train_data,
            transform=transform,
            num_workers=num_workers,
        )
        self.val_loader = megane_dataloader(
            val_data,
            transform=transform,
            num_workers=num_workers,
        )

    def run(self, action: str, **kwargs):
        if action == "train":
            self.train(**kwargs)
        else:
            raise ValueError(
                f"Unsupported action {action}, it's either train or test"
            )

    def train(self):
        train_loader = self.setup_dataloaders(self.train_loader)
        model, optimizer = self.setup(self.model, self.optimizer)

        # Stuffs from the context, avoid getattr calls
        lr_scheduler = self.lr_scheduler
        total_steps = self.total_steps
        validate_every = self.validate_every
        print_every = validate_every // 5
        criterion = self.criterion

        # Training with steps
        pbar = cycle(train_loader, total_steps)
        pbar = tqdm(pbar, "Training", total=total_steps, dynamic_ncols=True)
        for step, (images, annotations) in pbar:
            model.train()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(*outputs, *annotations)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if step % print_every == 0:
                lr = optimizer.param_groups[0]['lr']
                info = [
                    f"Step: {step}/{total_steps}",
                    f"Loss: {loss.item():.4f}",
                    f"Lr: {lr:.3e}",
                ]
                tqdm.write(" - ".join(info))

            if step % validate_every == 0:  # Step are 1 based
                metrics = self.validate()
                info = [f"{k}: {v}" for k, v in metrics.items()]
                tqdm.write(" - ".join(info))

    @torch.no_grad()
    def validate(self):
        val_loader = self.setup_dataloaders(self.val_loader)
        model = self.setup(self.model)
        criterion = self.criterion

        num_batches = len(val_loader)
        visualize_index = random.choice(range(num_batches))

        count = 0
        for images, annotations in val_loader:
            outputs = model(images)
            loss = criterion(*outputs, *annotations)

            # Check accuracy
            # The first map is the probability map
            proba_maps = torch.sigmoid(outputs[0])
            target_proba_maps = annotations[0]
            for proba_map, target_proba_map in zip(proba_maps, target_proba_maps):
                polygons, _, _ = self.post_processor(
                    proba_map.cpu().numpy()
                )
                target_polygons, _, _ = self.post_processor(
                    target_proba_map.cpu().numpy().astype('float32')
                )
                print(polygons, target_polygons)
                score = scores.f1_score(polygons, target_polygons)

            if visualize_index == count:
                proba_map = proba_maps[0][0].cpu().numpy()
                target_proba_map = torch.sigmoid(
                    proba_maps[0][0]).cpu().numpy()
                cv2.imshow("proba_map", proba_map)
                import numpy as np
                np.save("proba_map.npy", proba_map)
                np.save("proba_map.npy", proba_map)
                # cv2.imwrite("proba_map.jpg", proba_map)
                cv2.waitKey(25)

            count = count + 1

        # THIS IS A DRAFT
        return dict(f1score=score)
