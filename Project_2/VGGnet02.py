import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os


from proj2_helpers import make_predictions, load_test_imgs
from preprocessing import preprocess
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier

from metrics import *

PATH_training = "./data/training/"
PATH_test = "./data/test_set_images/"

IMG_WIDTH = 608
IMG_HEIGHT = 608
IMG_CHANNELS = 3
INPUT_SHAPE = ((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

ESTIMATORS = 50
RANDOM_STATE = 42

PATCH_SIZE = 16


class Vgg16 :

    def __init__(self, training_path = PATH_training, test_path = PATH_test, save_models = True, validation_set = False, predictions = False):
        """ Initialsizes a VGG16 net, train it and make predictions on the test set

            param PATH_training : path to the training data folder
            param PATH_test : path to the test data folder
            param save_models : if true, the models used for classification are saved as .h5 file
        """
        # load and preprocess train data :
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.RF_model = None

    def construct_existing_model(self):
        """ Use the VGG16 pretrained model for further analysis."""
        model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

        #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
        for layer in model.layers:
        	layer.trainable = False

        self.model = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)

        return self.model


    def construct_model(self, img_shape, include_top = False):
        """
            Initialsizes a VGG16 net.
        """

        self.model = Sequential()

        self.model.add(Conv2D(input_shape=INPUT_SHAPE,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

        """
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        """

        # Dense layer :
        if include_top:
            self.model.add(Flatten())

            self.model.add(Dense(units=4096,activation="relu"))
            self.model.add(Dense(units=4096,activation="relu"))
            self.model.add(Dense(units=1, activation="softmax")) # 1 output

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',
                                                                            f1_m,
                                                                            precision_m,
                                                                            recall_m])

        return self.model


    def save_models(self):
        """ Save the model in .h5 file for future use, preventing the re-compiling
            of the model each time
        """
        self.model.save('VGG16_net.h5')
        self.RF_model.save('RF_classifier.h5')


    def training(self,val_split = 0.1, batch = 16, epoch = 100):
        """ Training model."""

        callbacks = [
                ModelCheckpoint("vgg16_1.h5",
                                verbose=1,
                                monitor='val_acc',
                                save_best_only=True,
                                save_weights_only = False,
                                mode = 'auto',
                                period = 1),

                ReduceLROnPlateau (monitor='val_acc',
                                factor=0.4,
                                patience=5,
                                verbose=1,
                                mode='min',
                                min_lr=1e-7),

                EarlyStopping (monitor='val_acc',
                                min_delta = 0,
                                patience=20,
                                verbose = 1,
                                mode='min')]

        # model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=callbacks)
        self.model.fit(self.X_train, self.Y_train, validation_split=val_split, batch_size=batch, epochs = epoch, callbacks=callbacks)


    def classification(self, validation_set = False, estimators = ESTIMATORS, random_state = RANDOM_STATE, foreground_th = 0.55):
        """ Executes classification of Images using Random Forest."""

        # features= self.model.predict(self.X_train)
        features = self.model.predict(self.X_train)
        self.X_train = features.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

        #Reshape Y to match X
        self.Y_train = self.Y_train.reshape(-1)

        self.RF_model = RandomForestClassifier(n_estimators = estimators, random_state = random_state)
        self.RF_model.fit(self.X_train, self.Y_train)

        if validation_set:
            val_features = self.model.predict(self.X_val)
            self.X_val = val_feature.reshape(-1, val_features.shape[3])
            self.Y_val = self.Y_train.reshape(-1)

            results = self.RF_model.evaluate(X_val, Y_val)
            print(results)

        # self.Y_pred = RF_model.predict(self.X_test)

        return self.RF_model


    def predictions_RF(self, foreground_th = 0.55):
        "Make first predictions on test set"
        # load test data :
        self.X_test = load_test_imgs(PATH_test)

        #predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
        test_features = self.model.predict(self.X_test)
        self.X_test = test_feature.reshape(-1, test_features.shape[3])

        return self.X_test
