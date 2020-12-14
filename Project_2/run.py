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

NUM_FILTERS = 16
DEPTH = 3
ALPHA = 0.1
DROPOUT = 0.5
PATCH_SIZE = 16


def run_(train = False, use_VGG = False, use_fractal = False, save_imgs = False, upscale=False):
    #If upscaling is used, training images are padded with a mirror reflection.
    if(upscale): img_size=608
    else: img_size=400
    if(use_VGG):
        vgg = Vgg16(validation_set = True)
        vgg.construct_existing_model()
    else:
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
            X_train, X_test, Y_train, Y_test = preprocess(divide_set=True,
                                         save_imgs = save_imgs,
                                         upscale_to_test_size=upscale)

            model = train_model(X_train, Y_train, X_test, Y_test, img_size)

        imgs_test = load_test_imgs(TEST_IMGS_PATH)

        print("making predictions...")
        make_predictions(imgs_test, model, img_size, use_fractal = use_fractal)
        print("created submission")



def train_model(X_train, Y_train, X_test, Y_test, img_size):

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
    input_patch_size = Input((PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS))
    
    if use_fractal:
        model = build_fract_model(input_patch_size,
                                  filters = N_FILTERS,
                                  dropout = DROPOUT,
                                  depth=DEPTH,
                                  alpha = ALPHA,
                                  kernel_size = KERNEL_SIZE)
        
        _, X_train = img_to_patch(X_train)
        X_test_p, X_test = img_to_patch(X_test)
        _, Y_train = labels_to_hot_matrix(Y_train)
        Y_test_p, Y_test = labels_to_hot_matrix(Y_test)
        
    else:
        model = build_unet(input_size,
                           n_filters=N_FILTERS,
                           dropout_down=DROPOUT_DOWN,
                           dropout_up=DROPOUT_UP,
                           batch_norm=True,
                           activation_fct=ACTIV_FCT,
                           final_activation=FINAL_ACT,
                           kernel_size=KERNEL_SIZE,
                           dilate = DILATION)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy',
                            f1_m,
                            precision_m,
                            recall_m])

    model.fit(X_train, Y_train,
              validation_split=0.15,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=model_tools)

    # fitting the model to the train data
    # evaluating performance of the model
    print("evaluating performance of the model")
    print(model.evaluate(X_test, Y_test))
    return model
