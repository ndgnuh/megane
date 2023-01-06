import torch
from eyeball.models import backbones, heads
from eyeball import processor
from eyeball.loader import EyeballDataset
from icecream import install
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
install()

# features = torch.rand(1, 256, 256, 256)
# head = heads.Retina(in_channels=256, feature_size=256,
#                     num_classes=1, num_anchors=1)
# ic(head)
# with torch.no_grad():
#     boxes, _ = head(features)

# ic(boxes.shape)

pp = processor.RetinaProcessor()
data = EyeballDataset("data",
                      image_width=1024,
                      image_height=1024,
                      preprocess=pp.pre)
# ic(data)
image, box = data[0]
ic(image.size)
