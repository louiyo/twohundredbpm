import numpy as np
import tensorflow as tf
from UNet_model import build_unet, conv_batch, recall_m, precision_m, f1_m
from keras.layers import Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from proj2_helpers import create_submission, load_test_imgs, make_predictions
from preprocessing import *
from fractal_net import *

IMG_HEIGHT = IMG_WIDTH = 400
IMG_CHANNELS = 3
N_FILTERS = 32
DROPOUT_DOWN = 0.2
DROPOUT_UP = 0.2
ACTIV_FCT = 'relu'
FINAL_ACT = 'sigmoid'
KERNEL_SIZE = (3, 3)
EPOCHS = 100
MODEL_FILEPATH = './checkpoints/new_model.h5'
TEST_IMGS_PATH = './test_set_images/'
SUBMISSION_PATH = './submission/new_submission.csv'
BATCH_SIZE = 8
DILATION = True

NUM_FILTERS = 16
DEPTH = 3
ALPHA = 0.1
DROPOUT = 0.5
PATCH_SIZE = 16


def run_(train = False use_fractal = False):
    if(not train):
        input_size = Input((img_size, img_size, IMG_CHANNELS))
        input_patch_size = Input((PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS))
        if use_fractal:
            model = build_fract_model(input_patch_size,
                                      filters = NUM_FILTERS,
                                      dropout = DROPOUT,
                                      depth=DEPTH,
                                      alpha = ALPHA,
                                      kernel_size = KERNEL_SIZE)
        else: 
            model = build_unet(input_size,
                                n_filters=N_FILTERS,
                                dropout_down=DROPOUT_DOWN,
                                dropout_up=DROPOUT_UP,
                                batch_norm=True,
                                activation_fct=ACTIV_FCT,
                                final_activation=FINAL_ACT,
                                kernel_size=KERNEL_SIZE,
                                dilate=DILATION)
        model.load_weights('./checkpoints/bestmodel.h5')
        print('loaded weigths from ', 'bestmodel.h5')
    else:
        print('beginning training')
        X_train, X_test, Y_train, Y_test = preprocess()

        model = train_model(X_train, Y_train, X_test, Y_test)

    imgs_test = load_test_imgs(TEST_IMGS_PATH)

    print("making predictions...")
    make_predictions(imgs_test, model, use_fractal = use_fractal)
    print("created submission")



def train_model(X_train, Y_train, X_test, Y_test):
    
    print("Training shape = ", X_train.shape)

    cp = ModelCheckpoint(filepath=MODEL_FILEPATH,
                        verbose=1,
                        monitor='val_loss',
                        save_best_only=True)

    lr = ReduceLROnPlateau (monitor='val_loss',
                            factor=0.4,
                            patience=7,
                            verbose=1,
                            mode='min',
                            min_lr=1e-7)

    es = EarlyStopping (monitor='val_loss',
                        patience=15,
                        mode='min')

    model_tools = [cp,es,lr]

    input_size = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    unet_model = build_unet(input_size,
                            n_filters=N_FILTERS,
                            dropout_down=DROPOUT_DOWN,
                            dropout_up=DROPOUT_UP,
                            batch_norm=True,
                            activation_fct=ACTIV_FCT,
                            final_activation=FINAL_ACT,
                            kernel_size=KERNEL_SIZE,
                            dilate = DILATION)

    unet_model.compile(optimizer=Adam(lr=1e-4),
                       loss='binary_crossentropy',
                       metrics=['binary_accuracy',
                                f1_m])

    unet_model.fit(X_train, Y_train,
                   validation_split=0.15,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   callbacks=model_tools)

    # fitting the model to the train data
    # evaluating performance of the model
    print("evaluating performance of the model")
    print(unet_model.evaluate(X_test, Y_test))
    
    return unet_model
