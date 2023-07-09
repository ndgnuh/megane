import random
from functools import partial

import cv2
import numpy as np
import albumentations as A


def ChromaticAberration(px=(-10, 10), **kwargs):
    fn = partial(chromatic_aberration, px=px)
    return A.Lambda(image=fn, name="ChromaticAberration", **kwargs)


def chromatic_aberration(img, px=(-10, 10), **options):
    cs = cv2.split(img)
    h, w = img.shape[:2]
    mw, mh = w, h
    channels = []
    for i, chan in enumerate(cs):
        # sample scale
        dx = random.randint(px[0], px[1])
        dy = random.randint(px[0], px[1])

        # New dimension
        nw = w + dx
        nh = h + dy

        # Max dimension
        mw = max(mw, nw)
        mh = max(mh, nh)

        # Scale
        chan = cv2.resize(chan, (nw, nh))
        channels.append(chan)

    # Paste to target
    newimg = np.ones((mh, mw, 3), dtype="uint8")
    for i, chan in enumerate(channels):
        ch, cw = chan.shape[:2]
        newimg[:ch, :cw, i] = chan
    newimg = newimg[:mh, :mw]

    # Reszie to original
    newimg = cv2.resize(newimg, (w, h))
    return newimg
