import random
from datetime import datetime

import torch
import numpy as np
from torch import optim, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm
from lightning import Fabric
from tensorboardX import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from .models import Model
from .data import TextDetectionDataset
from .meanap import compute_maf1


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
    def __init__(self):
        image_size = 864
        assert image_size % 32 == 0
        train_data = "data/train.txt"
        val_data = "data/val.txt"
        classes = ["text", "noise"]
        dataloader_config = {"batch_size": 2}
        hidden_size = 256
        lr = 1e-4
        logdir = "log/expm-" + datetime.now().isoformat()
        total_steps = 1000000
        print_every = 100
        validate_every = 5
        self.latest_model_path = "latest.pt"

        #
        self.fabric = Fabric(accelerator="auto")

        # Model & optimization
        self.model = Model(
            {
                "image_size": image_size,
                "hidden_size": hidden_size,
                "num_classes": len(classes),
            }
        )
        self.model.load_state_dict(torch.load("best-model.pt", map_location='cpu'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # Dataloader
        def make_loader(data):
            data = TextDetectionDataset(
                data, classes, transform=self.model.encode_sample
            )
            return DataLoader(data, **dataloader_config)

        self.train_loader = make_loader(train_data)
        self.val_loader = make_loader(val_data)

        # Scheduling
        self.total_steps = total_steps
        self.print_every = print_every
        self.validate_every = validate_every

        # Logging
        self.logger = SummaryWriter(logdir=logdir)

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
            fabric.clip_gradients(model, optimizer, max_norm=5)
            optimizer.step()
            loss = loss.item()

            # Logging
            logger.add_scalar("train/loss", loss, step)
            pbar.set_postfix({"loss": loss})

            if step % print_every == 0:
                b_idx =  random.choice(range(images.shape[0]))
                sample = model.decode_sample(images[b_idx], outputs[b_idx])
                logger.add_image("train/sample", sample.visualize_tensor(), step)
                # logger.add_images(image)
                logger.flush()

            if step % validate_every == 0:
                self.validate(step)
                logger.flush()

                torch.save(model.state_dict(), "model.pt")
                torch.save(outputs.cpu().detach(), "sample-output.pt")

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
                maf1 = compute_maf1(
                    *pr_sample.adapt_metrics(),
                    *gt_sample.adapt_metrics()
                )
                maf1_set.append(maf1)

        # Logging
        # Visualize
        n = len(predictions)
        idx = random.randint(0, n - 1)
        logger.add_image("validate/sample-pr", predictions[idx].visualize_tensor(), step)
        logger.add_image("validate/sample-gt", ground_truths[idx].visualize_tensor(), step)

        # Metric
        logger.add_scalar(f"validate/mean-average-F1", np.mean(maf1_set), step)

        # Validation loss
        loss = np.mean(losses)
        logger.add_scalar("validate/loss", loss, step)


def _gen_image_preview(outputs: Tensor):
    """Create preview for logger

    Args:
        outputs:
            Tensor of shape [B, C, H, W]

    Returns:
        image:
            Tensor of shape [1, H, W]
    """
    # This actually gives much better results than sigmoid, softmax or thresholding
    outputs = torch.clip(torch.tanh(outputs), 0, 1)
    n = random.randint(0, outputs.shape[0] - 1)
    images = [image for image in outputs[n]]
    num_imgs = len(images)
    image = torch.cat(images, dim=-1)
    image = image.unsqueeze(0)
    image = TF.resize(image, (640, 640 * num_imgs), antialias=False)
    return image
