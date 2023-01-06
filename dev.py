from itertools import cycle
import torch
import icecream
# from pytorch_lightning import Trainer
from pytorch_lightning.lite import LightningLite
from eyeball.loader import EyeballDataset, DataLoader
from eyeball.models import backbones, heads, losses
from eyeball.models import detector
from eyeball import processor
from torch import optim, nn
from tqdm import tqdm
icecream.install()


class LossFunction(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "db":
            self.loss = losses.DBLoss()
        elif mode == "retina":
            self.loss = losses.RetinaLoss()
        else:
            raise ValueError("Unsupported loss mode " + str(mode))

    def forward(self, outputs, annotations):
        mode = self.mode
        if mode == 'db':
            prob_map, thres_map = outputs
            t_prob_map, t_thres_map = annotations
            return self.loss(prob_map, t_prob_map, thres_map, t_thres_map)
        else:
            raise ValueError("Unsupported loss mode " + str(mode))


class Trainer(LightningLite):
    def __init__(self, config, **k):
        super().__init__(**k)
        self.model = detector.Detector(config)
        self.processor = processor.DBProcessor()
        self.loss = LossFunction(config.mode)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     config.learning_rate)
        train_data = EyeballDataset(
            'data/',
            640,
            640,
            preprocess=self.processor.pre
        )
        val_data = EyeballDataset(
            'data/',
            640,
            640,
            preprocess=self.processor.pre
        )
        self.train_loader = DataLoader(train_data, batch_size=4)
        self.val_loader = DataLoader(train_data, batch_size=4)

        self.num_steps = 100

    def run(self):
        model, optimizer = self.setup(self.model, self.optimizer)
        train_loader = iter([])

        for step in tqdm(range(self.num_steps), "Training"):
            # Fetching the dataloader continously
            # First trainloader is set to empty to avoid
            # repetition
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = iter(self.setup_dataloaders(self.train_loader))
                batch = next(train_loader)

            # Forward pass
            image, annotations = batch
            optimizer.zero_grad()
            outputs = model(image)
            loss = self.loss(outputs, annotations)
            self.backward(loss)
            optimizer.step()


config = detector.Config(
    backbone="fpn_resnet18",
    mode="db",
    feature_size=256
)

# x = torch.rand(1, 3, 1024, 1024)
# with torch.no_grad():
#     y1, y2 = model.cuda()(x.cuda())
#     print(y1.shape)


trainer = Trainer(config, accelerator="gpu")
trainer.run()
