from torch.utils.data import DataLoader
from pytorch_lightning.lite import LightningLite
from tqdm import tqdm, trange
from os import path
from torch import optim
import torch

from . import models, loader, const, aug
from .tools import init
from .tools.meanap import calc_mean_ap


def cycle(total_steps, dataloader):
    step = 0
    while step <= total_steps:
        for batch in dataloader:
            step = step + 1
            yield step, batch


class Trainer(LightningLite):
    def __init__(self, model_config, train_config):
        super().__init__(accelerator="auto")
        # Configurations
        self.train_config = train_config
        self.model_config = model_config
        self.name = model_config['name']
        self.best_weight_path = path.join(
            const.weights_path,
            f"{self.name}-best.pt"
        )
        self.latest_weight_path = path.join(
            const.weights_path,
            f"{self.name}-latest.pt"
        )
        self.log_file = path.join(
            const.logs_path,
            f"{self.name}.log"
        )

        # Model initialization
        self.model = models.Detector(model_config)

        # Optimization & scheduling
        self.criterion = init.init_from_ns(
            models,
            model_config['loss'],
            model_config.get('loss_options', {}),
        )

        self.optimizer = init.init_from_ns(
            optim,
            train_config['optimizer'],
            train_config.get('optimizer_options', {}),
            self.model.parameters()
        )
        if 'lr_scheduler' in train_config:
            self.lr_scheduler = init.init_from_ns(
                optim.lr_scheduler,
                train_config['lr_scheduler'],
                train_config.get('lr_scheduler_options', {}),
                self.optimizer
            )

        # Iteration scheduling
        self.total_steps = train_config['total_steps']
        self.validate_every = train_config['validate_every']
        self.print_every = self.validate_every // 5

        # Prepare data
        self.train_loader = self.mk_dataloader(
            train_config['train_data'],
            augment=None,
        )
        self.val_loader = self.mk_dataloader(train_config['validate_data'])

    def mk_dataloader(self, data_root, augment=None):
        data = loader.MeganeDataset(
            data_root,
            self.model_config['image_width'],
            self.model_config['image_height'],
            preprocess=self.model.processor.pre,
            augment=augment,
        )

        dataloader_options = self.train_config['dataloader_options']
        dataloader_options.setdefault("shuffle", True)
        return DataLoader(data, **dataloader_options)

    def print(self, info):
        tqdm.write(info)
        with open(self.log_file, "a") as io:
            io.write(info)
            io.write("\n")

    def run(self, action: str = "train"):
        self.train()

    def train(self):
        model, optimizer = self.setup(self.model, self.optimizer)
        train_loader = self.setup_dataloaders(self.train_loader)

        criterion = self.criterion
        lr_scheduler = self.lr_scheduler

        # Avoid getattr calls
        validate_every = self.validate_every
        print_every = self.print_every
        backward = self.backward

        pbar = trange(1, self.total_steps + 1,
                      desc="Training", dynamic_ncols=True)
        for step, (images, labels) in cycle(self.total_steps, train_loader):
            pbar.update(1)

            # Train step
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            backward(loss)
            optimizer.step()
            lr_scheduler.step()

            if step % validate_every == 0:
                metrics = self.validate(model)
                info = [
                    f"Val loss: {metrics['loss']:3f}",
                    f"Score: {metrics['score']:3f}",
                ]
                self.print(" - ".join(info))

            if step % print_every == 0:
                lr = optimizer.param_groups[0]['lr']
                info = [
                    f"Step: {step}/{self.total_steps}",
                    f"Loss: {loss.item():.4f}",
                    f"Lr: {lr:.3e}",
                ]
                self.print(" - ".join(info))

    @torch.no_grad()
    def validate(self, model):
        val_loader = self.setup_dataloaders(self.val_loader)
        criterion = self.criterion
        processor = self.model.processor
        score = []
        losses = []
        for image, annotations in tqdm(val_loader, "Validating"):
            outputs = model(image)
            loss = criterion(outputs, annotations)
            prs = processor.post(outputs)
            gts = processor.post(annotations, is_target=True)
            meanap = calc_mean_ap(prs, gts)
            score.append(meanap)
            losses.append(loss)
        score = sum(score) / len(score)
        loss = sum(losses) / len(losses)
        return dict(score=score, loss=loss)
