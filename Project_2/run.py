import numpy as np
import tensorflow as tf
from UNet_model import build_unet, conv_batch
from metrics import f1_m
from keras.layers import Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from proj2_helpers import create_submission, load_test_imgs, make_predictions, img_to_patch, labels_to_hot_matrix
from preprocessing import *
from fractal_net import *
from metrics import *

IMG_HEIGHT = IMG_WIDTH = 400
IMG_CHANNELS = 3
N_FILTERS = 32
DROPOUT_DOWN = 0
DROPOUT_UP = 0.2
ACTIV_FCT = 'relu'
FINAL_ACT = 'sigmoid'
KERNEL_SIZE = (3, 3)
EPOCHS = 200
MODEL_FILEPATH = './checkpoints/new_model.h5'
TEST_IMGS_PATH = './test_set_images/'
SUBMISSION_PATH = './submission/new_submission.csv'
BATCH_SIZE = 8


# For Fractal Net:
DEPTH = 3
ALPHA = 0.1
DROPOUT = 0.5
PATCH_SIZE = 16


def run_(train = True, use_fractal = False, augment = True, augment_random = False, augment_factor = 7, dilation = True, display_preds = True):
    """
        Evaluation of the test images.

        :param train: if true, it will train the model
        :param use_fractal: if true, it will use the fractal model. By default it uses u_net.
        :param augment: if true, perform data augmentation when loading training data.
        :param augment_random: if true and augment is true as well, perform random data augmentation,
                               by default it will perform normal augmentation.
        :param augment_factor: if augment is true, it will perform augment_factor-fold augmentation.
        :param dilation: if True and using u-net, it will use dilation.
        :param display_preds: if True, it will display a few randomly chosen predictions.
    """
    if(not train):
        input_size = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        input_patch_size = Input((PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS))
        if use_fractal:
            model = build_fract_model(input_patch_size,
                                      filters = N_FILTERS,
                                      dropout = DROPOUT,
                                      depth = DEPTH,
                                      alpha = ALPHA,
                                      kernel_size = KERNEL_SIZE)
            model.load_weights('./checkpoints/Fractal_model.h5')
            print('loaded weigths from ', 'Fractal_model.h5')
        else:
            model = build_unet(input_size,
                                n_filters=N_FILTERS,
                                dilate=dilation,
                                dropout_down=DROPOUT_DOWN,
                                dropout_up=DROPOUT_UP,
                                batch_norm=True,
                                activation_fct=ACTIV_FCT,
                                final_activation=FINAL_ACT,
                                kernel_size=KERNEL_SIZE)
            model.load_weights('./checkpoints/bestmodel.h5')
            print('loaded weigths from ', 'bestmodel.h5')
    else:
        print('beginning training')
        X_train, X_test, Y_train, Y_test = preprocess(augment = augment,
                                                      augment_random = augment_random,
                                                      augment_factor = augment_factor)

        model,history = train_model(X_train, Y_train, X_test, Y_test,dilation ,use_fractal = use_fractal)

    imgs_test = load_test_imgs(TEST_IMGS_PATH)

    print("making predictions...")
    if use_fractal:
        get_prediction(imgs_test, model)
    else:
        make_predictions(imgs_test, model, display_preds = display_preds)
    print("created submission")
    if(train): return model, history

    


def train_model(X_train, Y_train, X_test, Y_test,dilation,use_fractal = False):

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
    input_patch_size = Input((PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS))

    if use_fractal:
        model = build_fract_model(input_patch_size,
                                  filters = N_FILTERS,
                                  dropout = DROPOUT,
                                  depth = DEPTH,
                                  alpha = ALPHA,
                                  kernel_size = KERNEL_SIZE)

        _, X_train = img_to_patch(X_train)
        _, X_test = img_to_patch(X_test)
        _, Y_train = labels_to_hot_matrix(Y_train)
        _, Y_test = labels_to_hot_matrix(Y_test)

    else:
        model = build_unet(input_size,
                           n_filters=N_FILTERS,
                           dilate=dilation,
                           dropout_down=DROPOUT_DOWN,
                           dropout_up=DROPOUT_UP,
                           batch_norm=True,
                           activation_fct=ACTIV_FCT,
                           final_activation=FINAL_ACT,
                           kernel_size=KERNEL_SIZE)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy',
                            f1_m])

    # fitting the model to the train data
    history = model.fit(X_train, Y_train,
                        validation_split=0.15,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=model_tools)

    # evaluating performance of the model
    print("evaluating performance of the model")
    print(model.evaluate(X_test, Y_test))

    return model, history
