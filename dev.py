import random
from megane.config import read_yaml, get_name
from argparse import ArgumentParser
from itertools import cycle
import torch
import icecream
# from pytorch_lightning import Trainer
from pytorch_lightning.lite import LightningLite
from megane.loader import EyeballDataset, DataLoader, default_augment
from megane.models import backbones, heads, losses
from megane.models import detector
from megane import processor
from megane.tools import meanap as MaP
from megane.tools import stats
from torch import optim, nn
from tqdm import tqdm
from argparse import Namespace
from megane.tools import init
from os import path
icecream.install()


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class Trainer(LightningLite):
    def __init__(self, model_config, train_config):
        lightning_options = train_config.get('lightning_options', {})
        super().__init__(**lightning_options)
        self.train_config = train_config
        self.model_config = model_config
        self.name = model_config['name']
        self.best_weight_path = path.join(
            "storage/weights/",
            f"{self.name}-best.pt"
        )
        self.latest_weight_path = path.join(
            "storage/weights/",
            f"{self.name}-latest.pt"
        )
        self.log_file = path.join(
            "storage/logs",
            f"{self.name}.log"
        )

        self.model = detector.Detector(model_config)

        self.loss = init.init_from_ns(
            losses,
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

        for k in ["total_steps", "print_every", "validate_every"]:
            setattr(self, k, train_config[k])

        self.train_loader = self.mk_dataloader(
            train_config['train_data'],
            augment=default_augment
        )
        self.val_loader = self.mk_dataloader(train_config['validate_data'])

    @torch.no_grad()
    def run_validate(self, model):
        val_loader = self.setup_dataloaders(self.val_loader)
        score = []
        losses = []
        for image, annotations in tqdm(val_loader, "Validating"):
            outputs = model(image)
            loss = self.loss(outputs, annotations)
            prs = self.model.processor.post(outputs)
            gts = self.model.processor.post(annotations, is_target=True)
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

        self.max_score = 0

        for self.global_step in tqdm(range(self.total_steps), "Training"):
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
            self.lr_scheduler.step()

            # Adversarial training
            image.requires_grad = True  # Collect input gradient

            # Generate adversarial example
            # model.eval() # TODO: DB doesn't generate threshold map in eval mode
            model.eval()
            aoutputs = model(image, head_options=dict(returns_threshold=True))
            aloss = self.loss(aoutputs, annotations)
            self.backward(aloss)
            data_grad = image.grad.data
            # epsilon = random.random() * 3
            epsilon = 0.1
            pimage = fgsm_attack(image, epsilon, data_grad)

            # Set the model to train mode
            model.train()
            optimizer.zero_grad()
            outputs = model(pimage)
            aloss = self.loss(outputs, annotations)
            self.backward(aloss)

            # Update
            optimizer.step()
            self.lr_scheduler.step()

            if self.is_matching_step(self.print_every):
                info = "[T] step: {step}/{total} - loss {loss:.4f} - lr {lr:.3e}"
                info = info.format_map({
                    "step": self.global_step,
                    "total": self.total_steps,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })
                self.print(info)

            # validation
            if self.is_matching_step(self.validate_every):
                metrics = self.run_validate(model)
                info = "[V] step: {step}/{total} - loss: {loss:.4f} - MeanAP: {meanap:.2f}"
                metrics['step'] = self.global_step
                metrics['total'] = self.total_steps
                info = info.format_map(metrics)
                self.print(info)
                self.save_weights(self.latest_weight_path)
                if self.max_score <= metrics['meanap']:
                    self.save_weights(self.best_weight_path)
                    self.max_score = metrics['meanap']

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
        self.print(f"Model saved to {path}")

    def print(self, info):
        tqdm.write(info)
        with open(self.log_file, "a") as io:
            io.write(info)
            io.write("\n")

    def mk_dataloader(self, data_root, augment=None):
        data = EyeballDataset(
            data_root,
            self.model_config['image_width'],
            self.model_config['image_height'],
            preprocess=self.model.processor.pre,
            augment=augment,
        )

        dataloader_options = self.train_config['dataloader_options']
        dataloader_options.setdefault("shuffle", True)
        return DataLoader(data, **dataloader_options)


class Predictor:
    def __init__(self, image_width: int, image_height: int):
        pass


# config = Namespace(
#     backbone="fpn_resnet18",
#     mode="db",
#     feature_size=256,
#     learning_rate=3e-4,
#     batch_size=4,
#     head_options=dict(head_size=256)
# )
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model-config", "-c",
                        required=True, dest="model_config")
    args = parser.parse_args()

    train_config = read_yaml("configs/train.yaml")
    model_config = read_yaml(args.model_config)
    trainer = Trainer(model_config, train_config)
    trainer.run()
