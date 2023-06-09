import traceback
import random
from os import path, makedirs
from datetime import datetime

import torch
import numpy as np
from torch import optim, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm
from lightning import Fabric
from tensorboardX import SummaryWriter

from .models import Model
from .data import TextDetectionDataset
from .meanap import compute_maf1
from .configs import TrainConfig, ModelConfig


def loop_loader(loader, total_steps: int):
    """Loop over dataloader for some steps

    Args:
        loader:
            The dataloader to loop over
        total_steps:
            Number of steps to loop

    Yields:
        step:
            The current step (starts from 1).
            This would makes it easier to implement actions every n steps without
            having to exclude the first step.
        batch:
            Whatever the dataloader yield.
    """
    step = 0
    while True:
        for batch in loader:
            step = step + 1
            yield step, batch
            if step == total_steps:
                return


class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        train_data = train_config.train_data
        val_data = train_config.val_data
        dataloader_config = train_config.dataloader
        fabric_config = train_config.fabric
        total_steps = train_config.total_steps
        print_every = train_config.print_every
        validate_every = train_config.validate_every
        lr = train_config.lr
        logdir = f"logs/{model_config.name}-{datetime.now().isoformat()}"

        weight_dir = "weights"
        self.best_weight_path = f"{weight_dir}/{model_config.best_weight_name}"
        self.latest_weight_path = f"{weight_dir}/{model_config.latest_weight_name}"

        # Torch fabric
        self.fabric = Fabric(**fabric_config)

        # Model & optimization
        self.model = Model(model_config)
        if model_config.continue_weight:
            self.model.load_state_dict(
                torch.load(model_config.continue_weight, map_location="cpu")
            )
        # self.model.load_state_dict(torch.load("best-model.pt", map_location='cpu'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # Dataloader
        def make_loader(data, **kwargs):
            data = TextDetectionDataset(
                data, model_config.classes, transform=self.model.encode_sample
            )
            return DataLoader(data, **dataloader_config, **kwargs)

        self.train_loader = make_loader(train_data, shuffle=True)
        self.val_loader = make_loader(val_data)
        tqdm.write(f"Num training batches: {len(self.train_loader)}")
        tqdm.write(f"Num validation batches: {len(self.val_loader)}")

        # Scheduling
        self.total_steps = total_steps
        self.print_every = print_every
        self.validate_every = validate_every

        # Logging
        self.logger = SummaryWriter(logdir=logdir)
        self.log_text("model", str(self.model), 0)

    def train(self):
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        dataloader = self.fabric.setup_dataloaders(self.train_loader)
        fabric = self.fabric

        logger = self.logger

        # scheduling
        total_steps = self.total_steps
        print_every = self.print_every
        validate_every = self.validate_every

        pbar = loop_loader(dataloader, total_steps)
        pbar = tqdm(pbar, "Training", total=total_steps)
        for step, (images, targets) in pbar:
            # Train step
            optimizer.zero_grad()
            outputs = model(images)
            loss = model.compute_loss(outputs, targets)
            fabric.backward(loss)
            # fabric.clip_gradients(model, optimizer, max_norm=5)
            optimizer.step()
            loss = loss.item()


            # Logging
            logger.add_scalar("train/loss", loss, step)
            pbar.set_postfix({"loss": loss})

            if step % print_every == 0:
                torch.save({"images": images[0], "pr": outputs[0], "gt": targets[0]}, "sample-output.pt")
                b_idx = random.choice(range(images.shape[0]))
                pr_sample = model.decode_sample(images[b_idx], outputs[b_idx])
                gt_sample = model.decode_sample(images[b_idx], targets[b_idx])
                logger.add_image("train/sample-pr", pr_sample.visualize_tensor(), step)
                logger.add_image("train/sample-gt", gt_sample.visualize_tensor(), step)
                logger.add_image(
                    "train/outputs-pr", model.head.visualize_outputs(outputs), step
                )
                logger.add_image(
                    "train/outputs-gt", model.head.visualize_outputs(targets), step
                )
                logger.flush()

            if step % validate_every == 0:
                self.validate(step)
                logger.flush()
                self.save_weight(self.latest_weight_path)

    def save_weight(self, savepath):
        dirname = path.dirname(savepath)
        makedirs(dirname, exist_ok=True)
        tqdm.write(f"Model weight saved to {savepath}")
        torch.save(self.model.state_dict(), savepath)

    @torch.no_grad()
    def validate(self, step=0):
        model = self.fabric.setup(self.model).eval()
        dataloader = self.fabric.setup_dataloaders(self.val_loader)
        num_batches = len(dataloader)

        logger = self.logger

        # Stats
        losses = []
        maf1_set = []
        predictions = []
        ground_truths = []
        raw_outputs = []
        raw_targets = []

        # Visualize index
        v_index = random.randint(0, num_batches - 1)
        pbar = tqdm(dataloader, "Validating")
        for idx, (images, targets) in enumerate(pbar):
            # Step
            outputs = model(images)

            # Loss
            loss = model.compute_loss(outputs, targets).item()
            losses.append(loss)

            # Inference output
            for _inputs, _outputs, _targets in zip(images, outputs, targets):
                pr_sample = model.decode_sample(_inputs, _outputs)
                gt_sample = model.decode_sample(_inputs, _targets * 1.0)
                predictions.append(pr_sample)
                ground_truths.append(gt_sample)
                raw_outputs.append(_outputs.cpu())
                raw_targets.append(_targets.cpu())
                maf1 = compute_maf1(
                    *pr_sample.adapt_metrics(), *gt_sample.adapt_metrics()
                )
                maf1_set.append(maf1)
                pbar.set_postfix({"loss": loss, "maf1": maf1})

        # Logging
        # Visualize
        n = len(predictions)
        idx = random.randint(0, n - 1)
        logger.add_image(
            "validate/sample-pr", predictions[idx].visualize_tensor(), step
        )
        logger.add_image(
            "validate/sample-gt", ground_truths[idx].visualize_tensor(), step
        )
        logger.add_image(
            "validate/outputs-pr", model.head.visualize_outputs(raw_outputs[idx]), step
        )
        logger.add_image(
            "validate/outputs-gt", model.head.visualize_outputs(raw_targets[idx]), step
        )

        # Metric
        logger.add_scalar(f"validate/mean-average-F1", np.mean(maf1_set), step)

        # Validation loss
        loss = np.mean(losses)
        logger.add_scalar("validate/loss", loss, step)
        logger.flush()

    def log_text(self, tag, txt, step):
        self.logger.add_text(tag, f"```\n{txt}\n```", step)
