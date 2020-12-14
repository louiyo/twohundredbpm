import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from mask_to_submission import *
import skimage.io as io

PATCH_SIZE = 16

# Helper functions


def load_test_imgs(path):
    test_imgs = []
    for i in range(1,51):
        test_img = img_to_array(load_img(os.path.join(path, "test_%d"%i, "test_%d.png"%i)))
        test_imgs.append(test_img)
    
    return np.array(test_imgs)

def predict(imgs,unet_model, use_fractal = False):
    width = 608
    height = 608
    predictions=[]
    for img in imgs:
        
        #dividing image in 4 different images
        img1 = img[:400, :400]
        img2 = img[:400, -400:]
        img3 = img[-400:, :400]
        img4 = img[-400:, -400:]
        
        #reshaping to meet model's tensor size input (1,400,400,3)
        img1 = np.reshape(img1,(1,)+img1.shape)
        img2 = np.reshape(img2,(1,)+img2.shape)
        img3 = np.reshape(img3,(1,)+img3.shape)
        img4 = np.reshape(img4,(1,)+img4.shape)
        
        if use_fractal:
            img1 = img_to_patch(img1)
            img2 = img_to_patch(img2)
            img3 = img_to_patch(img3)
            img4 = img_to_patch(img4)
        
        prediction = np.zeros((width, height, 1))
        prediction[:400, :400] = unet_model.predict(img1)
        prediction[:400, -400:] = unet_model.predict(img2)
        prediction[-400:, :400] = unet_model.predict(img3)
        prediction[-400:, -400:] = unet_model.predict(img4)
        predictions.append(prediction)
    
    return predictions



def make_predictions(imgs_test, model, img_size, name_of_csv = './submission/submission.csv', foreground_th = 0.55, use_fractal = False):

    if(img_size==400):
        imgs_preds = predict(imgs_test,model, use_fractal = use_fractal)
    elif(img_size==608): 
        if use_fractal:
            imgs_test = img_to_patch(imgs_test)
        imgs_pred = model.predict(np.asarray(imgs_test), batch_size = 1, verbose = 1)
    
  
    if not use_fractal:
        imgs_preds=np.asarray(imgs_preds)
        imgs_preds[imgs_preds <= foreground_th] = 0
        imgs_preds[imgs_preds > foreground_th] = 1
        print("ones",len(imgs_preds[imgs_preds==1]))
        print("zeros",len(imgs_preds[imgs_preds==0]))
        img_patches = [img_crop(img, PATCH_SIZE, PATCH_SIZE) for img in imgs_preds]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        preds = np.asarray([patch_to_label(np.mean(img_patches[i])) for i in range(len(img_patches))])
    else:
        preds = img_preds.copy()
    
    create_submission(preds, name_of_csv, img_size)
    
    

def create_submission(preds, name_of_csv, img_size):
    n = img_size // PATCH_SIZE
    preds = np.reshape(preds, (-1, n, n))
    
    with open(name_of_csv, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(preds)):
            img = preds[i]
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):    
                    f.write(('{:03d}_{}_{},{}'.format(i + 1, 
                                    j * PATCH_SIZE,
                                    k * PATCH_SIZE,
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

