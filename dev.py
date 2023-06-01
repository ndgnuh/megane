import torch
from src import Trainer
from matplotlib import pyplot as plt
from icecream import install
install()

Trainer().train()



# data = TextDetectionDataset("data.txt", ["text", "noise"])
# # print(data)

# image_size = 320
# assert image_size  % 32 == 0
# model = Model({
#     "image_size": image_size,
#     "hidden_size": 224,
#     "num_classes": 2,
# })
# # images = torch.rand(1, 3, image_size, image_size)
# # print(images.shape)
# # with torch.no_grad():
# #     outputs = model(images)
# #     print(outputs.shape)

# sample = data[0]
# image, target = model.encode_sample(sample)
# images = image.unsqueeze(0)
# targets = target.unsqueeze(0)
# # with torch.no_grad():
# outputs = model(images)
# print(outputs.shape)
# loss = model.compute_loss(outputs, targets)
# print(loss)
# # plt.imshow(targets[0])
# # plt.show()
