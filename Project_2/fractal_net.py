import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import BatchNormalization, MaxPooling2D, LeakyReLU, concatenate, Dropout, Conv2D, Flatten, Activation, add
from keras.optimizers import Adam
from mask_to_submission import patch_to_label
from proj2_helpers import img_crop

# INPUT = Input((400, 400, 3))
NUM_FILTER = 16
DROPOUT = 0.5
KERNEL_SIZE = (3,3)
ALPHA = 0.1
DEPTH = 3
PATCH_SIZE = 16

def labels_to_hot_matrix(labels):
    """
        Convert labels to one hot matrix.
    """
    gt_patches = [img_crop(gt, PATCH_SIZE, PATCH_SIZE) for gt in labels]
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    labels = np.asarray([patch_to_label(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    return labels.astype(np.float32)

def label_to_img(labels, width, height, patch_size = PATCH_SIZE):
    prediction = np.zeros([width, height])
    idx = 0
    for i in range(0,imgheight, patch_size):
        for j in range(0,imgwidth, patch_size):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            prediction[j:j+16, i:i+16] = l
            idx = idx + 1

    return prediction

def convBlock(img, filters = NUM_FILTER, alpha = ALPHA, kernel_size = KERNEL_SIZE):
    """
        Convolutional block with a convolution followed by a batch normalization
        and a leaky relu activation.
    """
    c1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(img)
    bn = BatchNormalization()(c1)
    act = LeakyReLU(alpha=alpha)(bn)

    return act

def fract_conv(img, filters = NUM_FILTER, depth=DEPTH, alpha = ALPHA, kernel_size = KERNEL_SIZE):
    """
    Create a fractal scruture in the model, like described on the report.
    """
    if (depth<=1):
        return convBlock(img, filters, alpha, kernel_size)

    else:
        c1 = convBlock(img, filters, alpha = alpha, kernel_size = kernel_size)
        c2 = fract_conv(img, filters, depth=depth-1, alpha = alpha, kernel_size = kernel_size)
        c3 = fract_conv(c2, filters, depth=depth-1, alpha = alpha, kernel_size = kernel_size)
        ct = concatenate([c1,c3])

        return ct

def build_fract_model(img, filters = NUM_FILTER, dropout = DROPOUT, depth=DEPTH, alpha = ALPHA, kernel_size = KERNEL_SIZE):
    """
        compute fractal model (clustering algorithm).
    """

    # 4 layer sctructure to go from a (16 x 16 x 3) input shape to (1 x 1 x 1)
    c1 = fract_conv(img, filters=filters, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p1 = MaxPooling2D((2,2))(c1)
    d1 = Dropout(dropout)(p1)

    c2 = fract_conv(d1, filters=filters*2, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p2 = MaxPooling2D((2,2))(c2)
    d2 = Dropout(dropout)(p2)

    c3 = fract_conv(d2, filters=filters*4, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p3 = MaxPooling2D((2,2))(c3)
    d3 = Dropout(dropout)(p3)

    c4 = fract_conv(d3, filters=filters*8, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p4 = MaxPooling2D((2,2))(c4)
    d4 = Dropout(dropout)(p4)

    c5 = Conv2D(2,(1,1), activation='sigmoid')(d4)
    outputs = Flatten()(c5)

    model = Model(inputs=[img], outputs=[outputs])
    model.summary()

    return model
