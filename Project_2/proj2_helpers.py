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


def predict(imgs, unet_model):
    '''make 4 400x400 predictions on 4 patches of the 608x608 test images
    the merge them together for a final prediction
    parameters:
    imgs:input test image
    unet_model: the unet model used to make the predictions
    '''
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

        prediction = np.zeros((width, height, 1))
        prediction[:400, :400] = unet_model.predict(img1)
        prediction[:400, -400:] = unet_model.predict(img2)
        prediction[-400:, :400] = unet_model.predict(img3)
        prediction[-400:, -400:] = unet_model.predict(img4)
        predictions.append(prediction)

    return predictions


def make_predictions(imgs_test, model, name_of_csv = './submission/submission.csv', foreground_th = 0.55, display_preds = True):
    
    imgs_preds = predict(imgs_test,model)
    imgs_preds=np.asarray(imgs_preds)
    
    if display_preds:
        for i in np.random.randint(0, 50, 4):
           
            cimg = concatenate_images(imgs_test[i], imgs_preds[i][:,:,0])
            fig1 = plt.figure(figsize=(10,10))
            plt.imshow(cimg, cmap = 'Greys_r')

    imgs_preds[imgs_preds <= foreground_th] = 0
    imgs_preds[imgs_preds > foreground_th] = 1
    print("ones",len(imgs_preds[imgs_preds==1]))
    print("zeros",len(imgs_preds[imgs_preds==0]))
    img_patches = [img_crop(img, PATCH_SIZE, PATCH_SIZE) for img in imgs_preds]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    preds = np.asarray([patch_to_label(np.mean(img_patches[i])) for i in range(len(img_patches))])

    create_submission(preds, name_of_csv)



def create_submission(preds, name_of_csv):
    n = 608 // PATCH_SIZE
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

def value_to_class(patch):
    """
        Return a list object for comparison with the output of fractal net.
    """
    df = np.mean(patch)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def img_to_patch(img):
    """
        Convert images to patch images. Patch are of size PATCH_SIZE x PATCH_SIZE.
    """
    img_patches = [img_crop(img_, PATCH_SIZE, PATCH_SIZE) for img_ in img]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return img, np.asarray(data)

def labels_to_hot_matrix(labels):
    """
        Convert labels to one hot matrix.
    """
    gt_patches = [img_crop(gt, PATCH_SIZE, PATCH_SIZE) for gt in labels]
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    labels = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    return np.rint(labels), labels.astype(np.float32)
