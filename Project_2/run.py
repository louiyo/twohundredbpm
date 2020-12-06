import numpy as np
import tensorflow as tf
from UNet_model import build_unet, conv_batch
from keras.layers import Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT = IMG_WIDTH = 400
IMG_CHANNELS = 3
N_FILTERS = 16
DROPOUT_DOWN = 0.05
DROPOUT_UP = 0.1
ACTIV_FCT = 'relu'
FINAL_ACT = 'sigmoid'
KERNEL_SIZE = (3, 3)
EPOCHS = 25
#STEPS_PER_EPOCH = 600
MODEL_NAME = 'new_model.h5'
SUBMISSION_PATH = './submission/new_submission.csv'
BATCH_SIZE = 16
DILATION=True

def run_(X_train, Y_train, X_test, Y_test):

    cp = ModelCheckpoint(MODEL_NAME, verbose=1,
                         monitor='val_loss', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                           patience=5, verbose=1, mode='min', min_lr=1e-7)
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_tools = [cp, lr, es]

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    unet_model = build_unet(inputs,
                            n_filters=16,
                            dropout_down=DROPOUT_DOWN,
                            dropout_up=DROPOUT_UP,
                            batch_norm=True,
                            activation_fct=ACTIV_FCT,
                            final_activation=FINAL_ACT,
                            kernel_size=KERNEL_SIZE,dilate=DILATION)

    unet_model.compile(optimizer=Adam(lr=1e-4),
                       loss='binary_crossentropy',
                       metrics='binary_accuracy')

    unet_model.fit(X_train, Y_train,
                   validation_split=0.1,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   callbacks=model_tools)

    # fitting the model to the train data
    # evaluating performance of the model
    results = unet_model.evaluate(X_test, Y_test)
    return results
