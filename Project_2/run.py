import numpy as np
import tensorflow as tf
from UNet_model import *
from keras.layers import Input, Lambda


IMG_HEIGHT = IMG_WIDTH = 400
IMG_CHANNELS = 3


def run(X_train, Y_train, X_test, Y_test):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    unet_model = build_unet(s, n_filters=16, dropout_down=0.05, dropout_up=0.1,
                            batchnorm=True, activation_fct='relu', final_activation='sigmoid', kernel_size=(3, 3))

    unet_model.fit(
        X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25)
    #results = unet_model.fit(X_train, Y_train, batch_size=16, epochs=25)

    # fitting the model to the train data
    # evaluating performance of the model
    result = unet_model.evaluate(X_test, Y_test)
    return result
