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
from scipy.ndimage import rotate
from skimage.util import random_noise
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf

def non_randaug(images,gt_images):
    '''
        Perform non random augmentations consisting of
        25,45,90,180,270 degree rotations with left to right image flips

        Parameters:
        img: satellite images
        get_img: respective groundtruth for each satellite image
    '''
    imgs_aug = []
    gt_imgs_aug = []

    for (img_, gt_img_) in zip(images, gt_images):

        imgs_aug.append(img_to_array(img_))
        gt_imgs_aug.append(img_to_array(gt_img_))

        #flipping
        img_flip = img_.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        gt_flip = gt_img_.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        imgs_aug.append(img_to_array(img_flip))
        gt_imgs_aug.append(img_to_array(gt_flip))

        #rotating
        img45=img_.rotate(45)
        gt45=gt_img_.rotate(45)
        imgs_aug.append(img_to_array(img45))
        gt_imgs_aug.append(img_to_array(gt45))

        img90=img_.rotate(90)
        gt90=gt_img_.rotate(90)
        imgs_aug.append(img_to_array(img90))
        gt_imgs_aug.append(img_to_array(gt90))

        img270=img_.rotate(270)
        gt270=gt_img_.rotate(270)
        imgs_aug.append(img_to_array(img270))
        gt_imgs_aug.append(img_to_array(gt270))

        img180=img_.rotate(180)
        gt180=gt_img_.rotate(180)
        imgs_aug.append(img_to_array(img180))
        gt_imgs_aug.append(img_to_array(gt180))


        img25=img_.rotate(25)
        gt25=gt_img_.rotate(25)
        imgs_aug.append(img_to_array(img25))
        gt_imgs_aug.append(img_to_array(gt25))

    return imgs_aug,gt_imgs_aug

def randaug(images, gt_images,augment_factor = 6):
    """Perform random augmentation on images and associated groundtruths."""
    rdaug = RandAugment(6, 12)

    imgs_aug = []
    gt_imgs_aug = []

    for (img_, gt_img_) in zip(images, gt_images):

        for j in range(augment_factor):
            img, gt_img = img_.copy(), gt_img_.copy()
            img, gt_img = rdaug.augment((img, gt_img))
            img, gt_img = img_to_array(img), img_to_array(gt_img)

            imgs_aug.append(img)
            gt_imgs_aug.append(gt_img)

    return imgs_aug, gt_imgs_aug


def add_noise(imgs, v):
    img = img_to_array(imgs[0])
    img_noise_arr = random_noise(img, mode='s&p',
                                 amount=v, seed=None, clip=False)
    return array_to_img(img_noise_arr), imgs[1]


def roll(original_imgs, delta):
    """Roll an image sideways."""

    delta = random.randint(75, 300)

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


def Rotate(imgs, _):
    v = random.randrange(0,180,1)
    img1 = array_to_img(rotate(img_to_array(imgs[0]), v, reshape=False))
    img2 = array_to_img(rotate(img_to_array(imgs[1]), v, reshape=False))
    return img1, img2

def AutoContrast(imgs, _):
    return PIL.ImageOps.autocontrast(imgs[0]), imgs[1]



def Flip(imgs, _):
    return PIL.ImageOps.mirror(imgs[0]), PIL.ImageOps.mirror(imgs[1])


def Contrast(imgs, v):  # [0.1,1.9]
    assert 0.5 <= v <= 1.0
    return PIL.ImageEnhance.Contrast(imgs[0]).enhance(v), imgs[1]


def Brightness(imgs, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(imgs[0]).enhance(v), imgs[1]


def Sharpness(imgs, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(imgs[0]).enhance(v), imgs[1]


def Identity(imgs, v):
    return imgs

def augment_list():  # Operations and their ranges
    return [
        (Identity, 0., 1.0),
        (Flip, 0, 1),
        (Rotate, 0, 180),
        (AutoContrast, 0, 1),
        (Contrast, 0.5, 1.0),
        (roll, 75, 300),
        (add_noise, 0.02, 0.025),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
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
            imgs = (img1, img2)

        return imgs[0], imgs[1]
