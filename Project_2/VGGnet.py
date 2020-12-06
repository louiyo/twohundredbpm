import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os


from proj2_helpers import make_predictions
from preprocessing import preprocess
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras.applications.vgg16 import VGG16

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier

PATH_training = "/Data/Training"
PATH_test = "/Data/test_set_images"

IMG_WIDTH = 400
IMG_HEIGHT = 400
IMG_CHANNELS = 3
INPUT_SHAPE = ((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

ESTIMATORS = 50
RANDOM_STATE = 42


class Vgg16 :

    def __init__(self, PATH_training, PATH_test, save_models = True, validation_set = False):
        """ Initialsizes a VGG16 net, train it and make predictions on the test set

            param PATH_training : path to the training data folder
            param PATH_test : path to the test data folder
            param save_models : if true, the models used for classification are saved as .h5 file
        """
        # load and preprocess train data :
        if validation_set:
            self.X_train, self.Y_train, self.X_val, self.Y_val = preprocess(root_dir=PATH_training, divide_set = validation_set)
        else :
            self.X_train, self.Y_train  = preprocess(root_dir=PATH_training)
        # load test data :
        self.X_test = load_test_set(PATH_test)
        # construct VGG16 net and train it :
        self.model = self.construct_model()
        self.training()

        # Use Random Forest to classify features given by VGG16 to make masks :
        self.RF_model = self.classification()

        # Predictions on Test data :
        make_predictions(self.X_test, self.RF_model)

        # Save models :
        if save_models == True:
            self.save_models()


    def construct_model(self):
        """
            Initialsizes a VGG16 net.
        """

        model = Sequential()

        model.add(Conv2D(input_shape=INPUT_SHAPE,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        # Dense layer :
        model.add(Flatten())

        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=1, activation="softmax")) # 1 output

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def save_models(self):
        """ Save the model in .h5 file for future use, preventing the re-compiling
            of the model each time
        """
        self.model.save('VGG16_net.h5')
        self.RF_model.save('RF_classifier.h5')


    def training(self, evaluation = False):
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

        # model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=callbacks)
        model.fit(self.X_train, self.Y_train, validation_split=0.1, batch_size=16, steps_per_epoch=100, callbacks=callbacks)


    # inutile dans notre cas !!!
    def remove_zero_labels(self):
        """ Remove the zero labels to reduce the dataset to classify."""
        #Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
        #In our labels Y values 0 = unlabeled pixels.
        dataset = pd.DataFrame(self.X_train)
        dataset['Label'] = self.Y_train
        print(dataset['Label'].unique())
        print(dataset['Label'].value_counts())

        ##If we do not want to include pixels with value 0
        ##e.g. Sometimes unlabeled pixels may be given a value 0.
        dataset = dataset[dataset['Label'] != 0]

        #Redefine X and Y for Random Forest
        self.X_train = dataset.drop(labels = ['Label'], axis=1)
        self.Y_train = dataset['Label']


    def classification(self, validation_set = False):
        """ Executes classification of Images using Random Forest."""

        features= self.model.predict(self.X_train)
        self.X_train = features.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

        #Reshape Y to match X
        self.Y_train = self.Y_train.reshape(-1)

        RF_model = RandomForestClassifier(n_estimators = ESTIMATORS, random_state = RANDOM_STATE)
        RF_model.fit(self.X_train, self.Y_train)

        #predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
        test_features = self.model.predict(self.X_test)
        self.X_test = test_feature.reshape(-1, test_features.shape[3])

        if validation_set:
            val_features = self.model.predict(self.X_val)
            self.X_val = val_feature.reshape(-1, val_features.shape[3])
            self.Y_val = self.Y_train.reshape(-1)

            results = RF_model.evaluate(X_val, Y_val)
            print(results)

        # self.Y_pred = RF_model.predict(self.X_test)

        return RF_model
