from itertools import cycle
import torch
import icecream
# from pytorch_lightning import Trainer
from pytorch_lightning.lite import LightningLite
from eyeball.loader import EyeballDataset, DataLoader
from eyeball.models import backbones, heads, losses
from eyeball.models import detector
from eyeball import processor
from eyeball.tools import meanap as MaP
from eyeball.tools import stats
from torch import optim, nn
from tqdm import tqdm
from argparse import Namespace
icecream.install()


class Trainer(LightningLite):
    def __init__(self, config, **k):
        super().__init__(**k)
        self.batch_size = config.batch_size
        self.mode = config.mode
        self.model = detector.Detector(config)
        self.processor = processor.DBProcessor()
        self.loss = losses.LossMixin(config.mode)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     config.learning_rate)
        self.train_loader = self.mk_dataloader('data')
        self.val_loader = self.mk_dataloader('data')
        self.validate_every = 200
        self.print_every = 200
        self.num_steps = 100000

    @torch.no_grad()
    def run_validate(self, model):
        val_loader = self.setup_dataloaders(self.val_loader)
        score = []
        losses = []
        for image, annotations in tqdm(val_loader, "Validating"):
            outputs = model(image)
            loss = self.loss(outputs, annotations)
            prs = self.processor.post(outputs)
            gts = self.processor.post(annotations)
            meanap = MaP.calc_mean_ap(prs, gts)
            score.append(meanap)
            losses.append(loss)
        score = sum(score) / len(score)
        loss = sum(losses) / len(losses)
        return dict(meanap=score, loss=loss)

    def is_matching_step(self, perstep):
        return self.global_step % perstep == 0 and self.global_step > 0

    def run(self):
        model, optimizer = self.setup(self.model, self.optimizer)
        train_loader = iter([])

        max_score = stats.MaxStatistic(value=-1)
        min_loss = stats.MinStatistic(value=10e10)

        for self.global_step in tqdm(range(self.num_steps), "Training"):
            # Fetching the dataloader continously
            # First trainloader is set to empty
            # to avoid repetition
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = iter(self.setup_dataloaders(self.train_loader))
                batch = next(train_loader)

            # Forward pass
            image, annotations = batch
            optimizer.zero_grad()
            model.train()
            outputs = model(image)
            loss = self.loss(outputs, annotations)
            self.backward(loss)
            optimizer.step()

            if self.is_matching_step(self.print_every):
                info = "[Train] step: {step}/{total} - loss {loss:.4f}"
                info = info.format_map({
                    "step": self.global_step,
                    "total": self.num_steps,
                    "loss": loss.item()
                })
                tqdm.write(info)

            # validation
            if self.is_matching_step(self.validate_every):
                metrics = self.run_validate(model)
                info = "[Validate] step: {step}/{total} - loss: {loss:.4f} - MeanAP: {meanap:.2f}"
                metrics['step'] = self.global_step
                metrics['total'] = self.num_steps
                info = info.format_map(metrics)
                tqdm.write(info)

    def mk_dataloader(self, data_root):
        data = EyeballDataset(
            data_root, 224, 224,
            preprocess=self.processor.pre
        )
        return DataLoader(data, batch_size=self.batch_size)


class Predictor:
    def __init__(self, image_width: int, image_height: int):
        pass


config = Namespace(
    backbone="fpn_resnet18",
    mode="db",
    feature_size=256,
    learning_rate=3e-4,
    batch_size=4,
)


trainer = Trainer(config, precision=16, accelerator="gpu")
trainer.run()
