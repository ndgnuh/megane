import torch
from src import TextDetectionDataset, Model
from matplotlib import pyplot as plt
from icecream import install
install()

data = TextDetectionDataset("data.txt", ["text", "noise"])
# print(data)

image_size = 320
assert image_size  % 32 == 0
model = Model({
    "image_size": image_size,
    "hidden_size": 224,
    "num_classes": 2,
})
# images = torch.rand(1, 3, image_size, image_size)
# print(images.shape)
# with torch.no_grad():
#     outputs = model(images)
#     print(outputs.shape)

sample = data[0]
image, targets = model.encode_sample(sample)
print(image.shape)
print(targets.dtype, targets.shape)
# plt.imshow(targets[0])
# plt.show()
