import numpy as np
import tensorflow as tf
from UNet_model import build_unet, conv_batch, recall_m, precision_m, f1_m
from keras.layers import Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from proj2_helpers import create_submission, load_test_imgs, make_predictions
from preprocessing import *

IMG_HEIGHT = IMG_WIDTH = 400
IMG_CHANNELS = 3
N_FILTERS = 32
DROPOUT_DOWN = 0.4
DROPOUT_UP = 0.4
ACTIV_FCT = 'relu'
FINAL_ACT = 'sigmoid'
KERNEL_SIZE = (3, 3)
EPOCHS = 200
MODEL_FILEPATH = './checkpoints/new_model.h5'
TEST_IMGS_PATH = './test_set_images/'
SUBMISSION_PATH = './submission/new_submission.csv'
BATCH_SIZE = 8
DILATION = True





def run_(train = False, use_VGG = False, save_imgs = False, upscale=False):
    #If upscaling is used, training images are padded with a mirror reflection.
    if(upscale): img_size=608
    else: img_size=400
    if(use_VGG):
        vgg = Vgg16(validation_set = True)
        vgg.construct_existing_model()
    else:
        if(not train):
            input_size = Input((img_size, img_size, IMG_CHANNELS))
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
            X_train, Y_train = preprocess(divide_set=False,
                                         save_imgs = save_imgs,
                                         upscale_to_test_size=upscale)

            model = train_model(X_train, Y_train, img_size)

        imgs_test = load_test_imgs(TEST_IMGS_PATH)

        print("making predictions...")
        make_predictions(imgs_test, model, img_size)
        print("created submission")



def train_model(X_train, Y_train, img_size):

    cp = ModelCheckpoint(filepath=MODEL_FILEPATH,
                        verbose=1,
                        monitor='val_loss',
                        save_best_only=True)

    lr = ReduceLROnPlateau (monitor='val_loss',
                            factor=0.4,
                            patience=5,
                            verbose=1,
                            mode='min',
                            min_lr=1e-7)

    es = EarlyStopping (monitor='val_loss',
                        patience=20,
                        mode='min')

    model_tools = [cp,es,lr]

    input_size = Input((img_size, img_size, IMG_CHANNELS))

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
                                f1_m,
                                precision_m,
                                recall_m])

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
