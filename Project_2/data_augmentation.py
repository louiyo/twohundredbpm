"""
    Data augmentation based on the work of Ekin D. Cubuk, 
    Barret Zoph, Jonathon Shlens, Quoc V. Le : 
    RandAugment: Practical automated data augmentation with 
    a reduced search space
    https://arxiv.org/abs/1909.13719
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from skimage.util import random_noise
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img



"""def add_noise(img, v):
    img_copy = img.copy()
    return random_noise(img_copy, mode='s&p', amount=ratio, seed=None, clip=True)
"""


def roll(original_imgs, delta):
    """Roll an image sideways."""
    def roll_single(img, delta):
        xsize, ysize = img.size
        part1 = img.crop((0, 0, delta, ysize))
        part2 = img.crop((delta, 0, xsize, ysize))
        img.paste(part1, (xsize-delta, 0, xsize, ysize))
        img.paste(part2, (0, 0, xsize-delta, ysize))
        return img
    assert 75 <= delta <= 300
    imgs = (original_imgs[0].copy(), original_imgs[1].copy())

    delta = delta % imgs[0].size[0]
    if delta == 0:
        return original_imgs

    return roll_single(imgs[0], int(delta)), roll_single(imgs[1], int(delta))


def Rotate(imgs, v):  # [-30, 30]
    if random.random() > 0.5:
        v = 90
    else:
        v = 180
    if random.random() > 0.5:
        v = -v
    return imgs[0].rotate(v), imgs[1].rotate(v)


def AutoContrast(imgs, _):
    return PIL.ImageOps.autocontrast(imgs[0]), imgs[1]


def Equalize(imgs, _):
    return PIL.ImageOps.equalize(imgs[0]), imgs[1]


def Flip(imgs, _):  # not from the paper
    return PIL.ImageOps.mirror(imgs[0]), PIL.ImageOps.mirror(imgs[1])


def Solarize(imgs, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(imgs[0], v), imgs[1]


def Posterize(imgs, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(imgs[0], v), imgs[1]


def Posterize2(imgs, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(imgs[0], v), imgs[1]


def Contrast(imgs, v):  # [0.1,1.9]
    assert 0.5 <= v <= 1.0
    return PIL.ImageEnhance.Contrast(imgs[0]).enhance(v), imgs[1]


def Brightness(imgs, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(imgs[0]).enhance(v), imgs[1]


def Sharpness(imgs, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(imgs[0]).enhance(v), PIL.ImageEnhance.Sharpness(imgs[1]).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(imgs, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return imgs
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return imgs


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(imgs, v):
    return imgs


def augment_list():  # 16 oeprations and their ranges
    return [
        (Identity, 0., 1.0),
        (Flip, 0, 1),
        (Rotate, 0, 180),  # 4
        (AutoContrast, 0, 1),  # 5
        (Contrast, 0.5, 1.0),
        (roll, 75, 300),
        (Posterize, 4, 8),  # 9
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
    ]


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def augment(self, imgs):
        # IMPORTANT : imgs should be a tuple containing
        # (satellite, ground_truth) <- the order matters!
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img1, img2 = op(imgs, val)

        return img1, img2
