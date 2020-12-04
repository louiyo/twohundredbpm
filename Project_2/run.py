import numpy as np
import tensorflow as tf
from UNet_model import build_unet, conv_batch
from keras.layers import Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from proj2_helpers import create_submission
from preprocessing import *

IMG_HEIGHT = IMG_WIDTH = 608
IMG_CHANNELS = 3
N_FILTERS = 16
DROPOUT_DOWN = 0.5
DROPOUT_UP = 0.5
ACTIV_FCT = 'relu'
FINAL_ACT = 'sigmoid'
KERNEL_SIZE = (3, 3)
EPOCHS = 20
#STEPS_PER_EPOCH = 400
MODEL_FILEPATH = './checkpoints/new_model.h5'
TEST_IMGS_PATH = './test_set_images/'
SUBMISSION_PATH = './submission/new_submission.csv'
BATCH_SIZE = 16


def run_(train = False):
    
    if(not train):
        input_size = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        model = build_unet(input_size,
                            n_filters=N_FILTERS,
                            dropout_down=DROPOUT_DOWN,
                            dropout_up=DROPOUT_UP,
                            batch_norm=True,
                            activation_fct=ACTIV_FCT,
                            final_activation=FINAL_ACT,
                            kernel_size=KERNEL_SIZE)
        model.load_weights('./checkpoints/bestmodel.h5')
        print('loaded weigths from ', MODEL_FILEPATH)
    else: 
        print('beginning training')
        X_train, X_test, Y_train, Y_test = preprocess(divide_set=True)
        #return X_train, X_test, Y_train, Y_test
        model = train_model(X_train, Y_train, X_test, Y_test)

    imgs_test = load_test_set(TEST_IMGS_PATH)
    
    create_submission(imgs_test, SUBMISSION_PATH)
    print("created submission")
    
    

def train_model(X_train, Y_train, X_test, Y_test):
    
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
                        
    model_tools = [cp, lr, es]

    input_size = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    unet_model = build_unet(input_size,
                            n_filters=N_FILTERS,
                            dropout_down=DROPOUT_DOWN,
                            dropout_up=DROPOUT_UP,
                            batch_norm=True,
                            activation_fct=ACTIV_FCT,
                            final_activation=FINAL_ACT,
                            kernel_size=KERNEL_SIZE)

    unet_model.compile(optimizer=Adam(lr=1e-4),
                       loss='binary_crossentropy',
                       metrics=['binary_accuracy'])

    unet_model.fit(X_train, Y_train,
                   validation_split=0.1,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   callbacks=[cp, lr, es])

    # fitting the model to the train data
    # evaluating performance of the model
    unet_model.evaluate(X_test, Y_test)
    return unet_model
