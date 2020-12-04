import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from mask_to_submission import *

PATCH_SIZE = 16
IMG_SIZE = 608

# Helper functions


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def load_test_imgs(path):
    test_imgs = []
    for image_dir in os.listdir(path):
        for img in os.listdir(image_dir):
            test_img = img_to_array(load_img(path + image_dir + img))
            test_imgs.append(test_img)
    test_imgs = np.round(np.stack(test_imgs, axis = 0) / 255)
    return test_imgs


def make_predictions(imgs_test, model, name_of_csv = 'submission.csv', foreground_th = 0.55):

    imgs_pred = model.predict(np.asarray(imgs_test), batch_size = 1, verbose = 1)
    imgs_pred[imgs_pred <= foreground_th] = 0
    imgs_pred[imgs_pred > foreground_th] = 1
    
    
    img_patches = [img_crop(img, PATCH_SIZE, PATCH_SIZE) for img in imgs_pred]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    preds = np.asarray([patch_to_label(np.mean(img_patches[i]), foreground_th) for i in range(len(img_patches))])
    
    create_submission(preds, name_of_csv)
    
    
def create_submission(preds, name_of_csv):
    n = IMG_SIZE // PATCH_SIZE
    y_pred = np.reshape(y_pred, (-1, n, n))
    
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(y_pred)):
            img = y_pred[i]
            for j in range(img.shape[0]):
                f.write(('{:03d}_{}_{},{}'.format(i + 1, 
                                    j * patch_size,
                                    k * patch_size,
                                    int(img[j,k])))+'\n')


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance


def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image


def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i])
                    for i in range(len(img_patches))])
    return X