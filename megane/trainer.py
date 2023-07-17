import math
import random
from datetime import datetime
from os import makedirs, path
from functools import partial

import numpy as np
import torch
from lightning import Fabric
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from megane.augment import Augmentation
from megane.configs import ModelConfig, TrainConfig, MeganeConfig
from megane.data import get_dataset
from megane.models import Model, ModelAPI
from megane.utils import compute_maf1, TimeoutException, time_limit, init_from_ns
from megane import registry
from megane.processors import get_processor


def generate_fgsm_example(model, images, targets):
    model.train()

    # Generating the fgsm attack
    delta = torch.zeros_like(images, device=images.device, requires_grad=True)
    outputs = model(images + delta)
    loss = model.compute_loss(outputs, targets)
    loss.backward()

    # Perturbation level
    epsilon = random.uniform(1 / 255, 8 / 255)
    delta = epsilon * delta.grad.detach().sign()

    # FGSM Example
    perturbed_images = torch.clamp(images + delta, 0, 1)
    return perturbed_images


def load_weights(model, weights):
    for name, params in model.named_parameters():
        if not name in weights:
            continue
        if weights[name].shape == params.shape:
            params.data = weights[name]


def batch_get_index(batch, i: int):
    if isinstance(batch, torch.Tensor):
        return batch[i]

    return [x[i] for x in batch]


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


def init_model(config: MeganeConfig, force_weights: bool = False):
    processor = init_from_ns(registry.processors, config.input_processor)
    encoder = init_from_ns(registry.target_encoders, config.target_encoder)
    decoder = init_from_ns(registry.target_decoders, config.target_decoder)
    model = Model(config)

    loaded_weight = False
    for weight in config.weights:
        if not path.isfile(weight):
            print("Weight {weight} not found, skipping loading it")
            continue

        weight = torch.load(weight, map_location="cpu")
        load_weights(model, weight)
        loaded_weight = True

    if force_weights:
        assert loaded_weight, "No weights were loaded"

    return model, processor, encoder, decoder


class Trainer:
    def __init__(self, config: MeganeConfig):
        train_config = config.train_config
        train_data = train_config.train_data
        val_data = train_config.val_data
        dataloader_config = train_config.dataloader
        fabric_config = train_config.fabric
        total_steps = train_config.total_steps
        print_every = train_config.print_every
        validate_every = train_config.validate_every
        lr = train_config.lr
        logdir = f"logs/{config.name}-{datetime.now().isoformat()}"

        weight_dir = "weights"
        self.best_weight_path = f"{weight_dir}/{config.best_weight_name}"
        self.latest_weight_path = f"{weight_dir}/{config.latest_weight_name}"

        # Preprocessor
        self.model, self.preprocess, self.encode, self.decode = init_model(config)

        # Torch fabric
        self.fabric = Fabric(**fabric_config)

        # Model & optimization
        # self.model.load_state_dict(torch.load("best-model.pt", map_location='cpu'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = ConsineDecayWithWarmup(
            self.optimizer,
            total_steps=train_config.total_steps,
            num_wramup_steps=30,
            min_pct=0.1,
        )

        # Dataloader
        augment = train_config.augment
        augment_enabled = augment.enabled
        if augment_enabled:
            augmentation = Augmentation(
                prob=augment.prob,
                background_images=augment.background_images,
                domain_images=augment.domain_images,
            )

        def make_loader(data, augment: bool, **kwargs):
            # Transform function
            def transform(sample):
                sample = self.preprocess(sample)
                if augment:
                    sample = augmentation(sample)
                enc = self.encode(sample)
                return enc

            # Dataset
            print("[trainer.py: 149] TODO: configurable num class for dataset")
            data = get_dataset(data, transform=transform, **train_config.data_options)

            # Datalodaer
            loader = DataLoader(
                data,
                **dataloader_config,
                **kwargs,
                collate_fn=self.model.collate_fn,
            )
            return loader

        self.train_loader = make_loader(
            train_data,
            shuffle=True,
            augment=augment_enabled,
        )
        self.val_loader = make_loader(val_data, augment=False)
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
        lr_scheduler = self.lr_scheduler
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
            optimizer.zero_grad()
            if random.choice((True, False)):
                images = generate_fgsm_example(model, images, targets)

            # Train step
            optimizer.zero_grad()
            outputs = model(images, targets)
            loss = model.compute_loss(outputs, targets)
            fabric.backward(loss)
            # fabric.clip_gradients(model, optimizer, max_norm=5)
            optimizer.step()
            lr_scheduler.step()
            loss = loss.item()

            # Logging
            lr = lr_scheduler.get_last_lr()[0]
            logger.add_scalar("train/lr", lr, step)
            logger.add_scalar("train/loss", loss, step)
            pbar.set_postfix({"loss": loss})

            if step % print_every == 0:
                # torch.save(
                #     {"images": images[0], "pr": outputs[0], "gt": targets[0]},
                #     "sample-output.pt",
                # )
                b_idx = random.choice(range(images.shape[0]))
                self.save_weight(self.latest_weight_path)

                # Model activate visualization
                model.visualize_outputs(
                    outputs,
                    logger=logger,
                    tag="train/outputs-pr",
                    step=step,
                    ground_truth=False,
                )
                model.visualize_outputs(
                    targets,
                    logger=logger,
                    tag="train/outputs-gt",
                    step=step,
                    ground_truth=True,
                )

                # Decode and visualize ground truth
                gt_sample = self.decode(
                    batch_get_index(images, b_idx),
                    batch_get_index(targets, b_idx),
                    ground_truth=True,
                )
                logger.add_image(
                    "train/sample-gt",
                    gt_sample.visualize_tensor(),
                    step,
                )

                # Flush while we wait for the sampe prediction decode
                logger.flush()

                try:
                    # Put on a timer because they tends to get really long at first
                    with time_limit(5):
                        # Decode and visualize prediction
                        pr_sample = self.decode(
                            batch_get_index(images, b_idx),
                            batch_get_index(outputs, b_idx),
                        )
                        logger.add_image(
                            "train/sample-pr",
                            pr_sample.visualize_tensor(),
                            step,
                        )
                except TimeoutException:
                    pass
                logger.flush()

            if step % validate_every == 0:
                self.validate(step)
                model = model.train()
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
            outputs = model(images, targets)

            # Loss
            loss = model.compute_loss(outputs, targets).item()
            losses.append(loss)

            # To CPU before decoding to avoid CUDA OOM
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu()
            elif isinstance(outputs, (tuple, list)):
                outputs = [out.cpu() for out in outputs]
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu()
            elif isinstance(targets, (tuple, list)):
                targets = [out.cpu() for out in targets]
            images = images.cpu()

            # Inference output
            for i in range(images.shape[0]):
                _inputs = batch_get_index(images, i)
                _outputs = batch_get_index(outputs, i)
                _targets = batch_get_index(targets, i)

                pr_sample = self.decode(_inputs, _outputs)
                gt_sample = self.decode(_inputs, _targets, ground_truth=True)
                predictions.append(pr_sample)
                ground_truths.append(gt_sample)
                raw_outputs.append(_outputs)
                raw_targets.append(_targets)
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
        model.visualize_outputs(
            raw_outputs[idx],
            logger=logger,
            tag="validate/outputs-pr",
            step=step,
            ground_truth=False,
        )
        model.visualize_outputs(
            raw_targets[idx],
            logger=logger,
            tag="validate/outputs-gt",
            step=step,
            ground_truth=True,
        )

        # Metric
        logger.add_scalar("validate/mean-average-F1", np.mean(maf1_set), step)

        # Validation loss
        loss = np.mean(losses)
        logger.add_scalar("validate/loss", loss, step)
        logger.flush()

    def log_text(self, tag, txt, step):
        self.logger.add_text(tag, f"```\n{txt}\n```", step)


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations, min_pct):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (
            total_iterations - warmup_iterations
        )
        multiplier = multiplier * (1 - min_pct)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def ConsineDecayWithWarmup(optimizer, num_wramup_steps, total_steps, min_pct):
    schedule = partial(
        _cosine_decay_warmup,
        warmup_iterations=num_wramup_steps,
        total_iterations=total_steps,
        min_pct=min_pct,
    )
    return lr_scheduler.LambdaLR(optimizer, schedule)
