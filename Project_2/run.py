import numpy as np
import tensorflow as tf
import UNet_model
from keras import layers
from layers import Input,Lambda







inputs =Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

unet_model=UNet_model.build_unet(input=s,n_filters=16,dropout_down=0.05,dropout_up=0.1,\
batchnorm=True,activation_fct='relu',final_activation='sigmoid',kernel_size=(3,3))

results = unet_model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25)
#results = unet_model.fit(X_train, Y_train, batch_size=16, epochs=25)

#fitting the model to the train data
unet_model.fit(x_train,y_train,epochs=16,batch_size=32)
#evaluating performance of the model
unet_model.evaluate(x_test,y_test)
