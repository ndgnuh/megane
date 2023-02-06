import sys
import lmdb
import json
from tqdm import tqdm
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    sys.path.append(os.curdir)
    from megane import loaders
    dataset = loaders.MeganeDataset(
        index="toybox/word-level-megane/index.txt",
        transform=None,
    )
    loaders.create_lmdb_dataset("toybox/word-level-lmdb/", dataset)
