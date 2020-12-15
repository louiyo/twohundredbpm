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


def random_crop(imgs, v):
    crop_size = int( 16 * round(v*imgs[0].width / 16))
    size = tf.constant([crop_size, crop_size], dtype=tf.int32)
    image, gt = img_to_array(imgs[0]), img_to_array(imgs[1])

    combined = tf.concat([image, gt], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
        combined, 0, 0, image_shape[0], image_shape[1])
    last_label_dim = tf.shape(gt)[-1]
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.image.random_crop(
        combined_pad,
        size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
    img_cropped_1 = array_to_img(combined_crop[:, :, :last_image_dim])
    img_cropped_2 = array_to_img(combined_crop[:, :, last_image_dim:])
    return img_cropped_1, img_cropped_2


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
    def __init__(self, n, m, upscale_to_test_size = False):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()
        if(upscale_to_test_size):
            self.augment_list.append((random_crop, 0.6, 0.9))

    def augment(self, imgs):
        # IMPORTANT : imgs should be a tuple containing
        # (satellite, ground_truth) <- the order matters!
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img1, img2 = op(imgs, val)
            imgs = (img1, img2)

        return imgs[0], imgs[1]
