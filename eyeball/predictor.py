from . import ops
from .models.detector import Detector
from . import config as configs
from torchvision.transforms import functional as TF
import torch


class Predictor:
    def __init__(self, config):
        self.model = Detector(config)
        self.image_width = config['image_width']
        self.image_height = config['image_height']

    @classmethod
    def from_config(cls, config_path: str):
        config = configs.read_yaml(config_path)
        configs.validate_config(config, configs.MODEL_CONFIG)
        return cls(config)

    def eval(self):
        self.model = self.model.eval()

    def train(self):
        self.model = self.model.train()

    def predict_single(self, image, return_maps=False, **options):
        width, height = image.size
        image = ops.preprocess_image(
            image,
            self.image_width,
            self.image_height
        )
        # To batch
        image = TF.to_tensor(image).unsqueeze(0)
        logits, threshold_maps = self.model(image)
        proba_maps = torch.sigmoid(logits * 50)

        # Unbatch and to numpy
        proba_maps = proba_maps.squeeze(0)
        proba_maps = proba_maps.cpu().detach().numpy()

        # Each classes
        results = []
        for (class_idx, proba_map) in enumerate(proba_maps):
            boxes, scores = ops.mask_to_boxes(
                proba_map,
                **options
            )
            results_ = [
                {
                    "box": ops.denormalize(
                        box,
                        width=width,
                        height=height,
                        norm_constant=1
                    ),
                    "score": score,
                    "class": class_idx
                }
                for box, score in zip(boxes, scores)
            ]
            results.extend(results_)

        if return_maps:
            results = (results, (logits, threshold_maps))
        return results

    def visualize_result(self, image, results, color=(255, 0, 0), stroke=2):
        image = image.copy()
        boxes = [tuple(map(int, r['box'])) for r in results]
        return ops.draw_rects(image, boxes, outline=color, width=stroke)
