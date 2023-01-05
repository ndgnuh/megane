from eyeball.loader import EyeballDataset
from eyeball.processor import DBProcessor

processor = DBProcessor()
d = EyeballDataset("data/", preprocess=processor.pre)
print(d)
print(d[0])
