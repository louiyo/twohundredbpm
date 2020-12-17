import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import BatchNormalization, MaxPooling2D, LeakyReLU, concatenate, Dropout, Conv2D, Flatten, Activation, add
from keras.optimizers import Adam
from mask_to_submission import patch_to_label
from proj2_helpers import img_crop, create_submission

# INPUT_SIZE = Input((400, 400, 3))
NUM_FILTER = 16
DROPOUT = 0.5
KERNEL_SIZE = (3,3)
ALPHA = 0.1
DEPTH = 3
PATCH_SIZE = 16

foreground_th = 0.55


def get_prediction(imgs, model, name_of_csv = './submission/submission.csv'):
    """
        Make the prediction of the fractal model.

        :param imgs: images, should be the test images (without patching).
        :param model: model which predict the groundtruth on patched images.
        :param name_of_csv: name of the csv for submission (plus the path).
    """
    # Convert images to patch:
    img_patches = [img_crop(img, PATCH_SIZE, PATCH_SIZE) for img in imgs]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    # Make Prediction with model:
    imgs_preds = model.predict(img_patches)

    imgs_preds_ = np.zeros(len(imgs_preds))

    # Assign values 0 or 1 to each patch depending on the highest probability:
    for i in range(len(imgs_preds)):
      if imgs_preds[i][0] >= imgs_preds[i][1]:
        imgs_preds_[i] = 0
      else :
        imgs_preds_[i] = 1

    print("ones",len(imgs_preds_[imgs_preds_==1]))
    print("zeros",len(imgs_preds_[imgs_preds_==0]))

    create_submission(imgs_preds_, name_of_csv)


def label_to_img(labels, width, height, patch_size = PATCH_SIZE):
    """
        Reconstruct the groundtruth image from the values of the patched image (not used here).

        :param labels: prediction of the patched images.
        :param width: width of the original image.
        :param height: height of the original image (should be the same as height).
        :param patch_size: size of the patch (group of pixels of size patch_size x patch_size).
    """
    prediction = np.zeros([width, height])
    idx = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if labels[idx][0] > foreground_th:
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

        :param img: input_size of the model.
        :param filters: number of filters to be used.
        :param alpha: parameter for function LeakyReLU (in convBlock).
        :param kernel_size: size of the kernel for Conv2D operations.
    """
    c1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(img)
    bn = BatchNormalization()(c1)
    act = LeakyReLU(alpha=alpha)(bn)

    return act

def fract_conv(img, filters = NUM_FILTER, depth=DEPTH, alpha = ALPHA, kernel_size = KERNEL_SIZE):
    """
        Create a fractal scruture in the model, like described on the report.

        :param img: input_size of the model.
        :param filters: number of filters to be used.
        :param depth: depth of each block of convolutional layer (in fract_conv).
        :param alpha: parameter for function LeakyReLU (in convBlock).
        :param kernel_size: size of the kernel for Conv2D operations.
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
        compute fractal model, which is a sequence of 4 blocks linked together by pooling operations.

        :param img: input_size of the model.
        :param filters: number of filters to be used.
        :param dropout: parameter for dropout function.
        :param depth: depth of each block of convolutional layer (in fract_conv).
        :param alpha: parameter for function LeakyReLU (in convBlock).
        :param kernel_size: size of the kernel for Conv2D operations.
    """
    # Block 1:
    c1 = fract_conv(img, filters=filters, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p1 = MaxPooling2D((2,2))(c1)
    d1 = Dropout(dropout)(p1)

    # Block 2:
    c2 = fract_conv(d1, filters=filters*2, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p2 = MaxPooling2D((2,2))(c2)
    d2 = Dropout(dropout)(p2)

    # Block 3:
    c3 = fract_conv(d2, filters=filters*4, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p3 = MaxPooling2D((2,2))(c3)
    d3 = Dropout(dropout)(p3)

    # Block 4:
    c4 = fract_conv(d3, filters=filters*8, depth = depth, alpha = alpha, kernel_size = kernel_size)
    p4 = MaxPooling2D((2,2))(c4)
    d4 = Dropout(dropout)(p4)

    c5 = Conv2D(2,(1,1), activation='sigmoid')(d4)
    outputs = Flatten()(c5)

    model = Model(inputs=[img], outputs=[outputs])
    model.summary()

    return model
