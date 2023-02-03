import cv2
import torch
import random
from os import path
from pytorch_lightning.lite import LightningLite
from torch import optim
from tqdm import tqdm

from . import stats
from .configs import read_config
from .augments import Augment
from .loaders import megane_dataloader
from . import transforms, losses, models, scores


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
        model_config: str,
        train_data: str,
        val_data: str,
        total_steps: int = 10_000,
        validate_every: int = 100,
        learning_rate: float = 3e-4,
        batch_size: int = 4,
        num_workers: int = 1,
    ):
        super().__init__(accelerator="auto")
        self.name = path.splitext(path.basename(model_config))[0]
        self.best_weight_path = path.join(
            "storage", "weights", f"{self.name}.pt"
        )
        self.model_config = read_config(model_config)
        self.total_steps = total_steps
        self.validate_every = validate_every

        # Model
        self.criterion = losses.DBLoss()
        self.model = models.DBNet.from_config(self.model_config)
        self.post_processor = transforms.DBPostprocess.from_config(
            self.model_config
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        # datasets
        transform = transforms.DBPreprocess.from_config(self.model_config)

        self.train_loader = megane_dataloader(
            train_data,
            transform=transforms.Compose([
                Augment.from_string("yes"),
                transform
            ]),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = megane_dataloader(
            val_data,
            transform=transform,
            batch_size=batch_size,
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

        # Metrics
        avg_train_loss = stats.AverageStatistic()
        max_score = stats.MaxStatistic()

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

            avg_train_loss.append(loss.item())
            if step % print_every == 0:
                lr = optimizer.param_groups[0]['lr']

                info = [
                    f"Step: {step}/{total_steps}",
                    f"Loss: {avg_train_loss.summarize():.4f}",
                    f"Best: {max_score.summarize():.3f}",
                    f"Lr: {lr:.3e}",
                ]
                tqdm.write(" - ".join(info))

            if step % validate_every == 0:  # Step are 1 based
                metrics = self.validate()
                changed = max_score.append(metrics['score'])
                info = [f"{k}: {v}" for k, v in metrics.items()]
                tqdm.write(" - ".join(info))
                if changed:
                    torch.save(model.state_dict(), self.best_weight_path)
                    tqdm.write(f"Best model saved to {self.best_weight_path}")

    @torch.no_grad()
    def validate(self):
        val_loader = self.setup_dataloaders(self.val_loader)
        model = self.setup(self.model)
        criterion = self.criterion

        num_batches = len(val_loader)
        visualize_index = random.choice(range(num_batches))

        avg_f1_score = stats.AverageStatistic()
        avg_val_loss = stats.AverageStatistic()

        count = 0
        for images, annotations in tqdm(val_loader, "Validating", dynamic_ncols=True):
            outputs = model(images)
            loss = criterion(*outputs, *annotations)
            avg_val_loss.append(loss.item())

            # Check accuracy
            # The first map is the probability map
            proba_maps = torch.sigmoid(outputs[0])
            target_proba_maps = annotations[0]
            for proba_map, target_proba_map in zip(proba_maps, target_proba_maps):
                polygons, _, _, _ = self.post_processor(
                    proba_map.cpu().numpy()
                )
                target_polygons, _, _, _ = self.post_processor(
                    target_proba_map.cpu().numpy().astype('float32')
                )
                avg_f1_score.append(scores.f1_score(polygons, target_polygons))

            if visualize_index == count:
                proba_map = proba_maps[0][0].cpu().numpy()
                target_proba_map = torch.sigmoid(
                    proba_maps[0][0]).cpu().numpy()
                cv2.imshow("proba map", proba_map)
                cv2.waitKey(1)

            count = count + 1

        # THIS IS A DRAFT
        return dict(
            score=avg_f1_score.summarize(),
            loss=avg_val_loss.summarize()
        )
