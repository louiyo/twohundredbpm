import tensorflow as tf
import numpy as np
from tf import keras
from keras import Model
from keras.layers import Activation, BatchNormalization, MaxPooling2D, concatenate, Dropout,Conv2D, Conv2DTranspose
from keras.optimizers import Adam


def conv_batch(input,n_filters,kernel_size=(3,3),batch_norm,activation_fct):

    #Maybe try to add dropout between convolutions instead of outside

    #first convolution followed by batch normalization
    c1=Conv2D(n_filters,kernel_size,kernel_initializer='he_normal',padding='same')(input)

    if (batch_norm):
        c1=BatchNormalization(c1)
    c1=Activation(activation_fct)(c1)

    #2nd convolution followed by batch normalization
    c2=Conv2D(n_filters,kernel_size,kernel_initializer='he_normal',padding='same')(c1)

    if (batch_norm):
        c2=BatchNormalization(c2)
    c2=Activation(activation_fct)(c2)

    return c2


def build_unet(input,n_filters,dropout_down=0.0,dropout_up=0.0,batchnorm=True,activation_fct='relu',final_activation='sigmoid',kernel_size=(3,3)):

    #downwards path
    c1=conv_batch(input,n_filters,kernel_size,batch_norm,activation_fct)
    p1=MaxPooling2D((2,2))(c1)
    d1=Dropout(dropout_down)(p1)

    c2=conv_batch(d1,n_filters*2,kernel_size,batch_norm,activation_fct)
    p2=MaxPooling2D((2,2))(c2)
    d2=Dropout(dropout_down)(p2)

    c3=conv_batch(d2,n_filters*4,kernel_size,batch_norm,activation_fct)
    p3=MaxPooling2D((2,2))(c3)
    d3=Dropout(dropout_down)(p3)

    c4=conv_batch(d3,n_filters*8,kernel_size,batch_norm,activation_fct)
    p4=MaxPooling2D((2,2))(c4)
    d4=Dropout(dropout_down)(p4)

    #lateral path
    c5=conv_batch(d4,n_filters*16,kernel_size,batch_norm,activation_fct)



    #upwards path (try dropout after conv_batch)
    up6=Conv2DTranspose(n_filters*8,kernel_size,strides=(2,2),padding='same',kernel_initializer='he_normal')
    up6=concatenate([up6,c4])
    up6=Dropout(dropout_up)(up6)
    c6=conv_batch(up6,n_filters*8,kernel_size,batch_norm,activation_fct)
    #c6=Dropout(dropout_up)(c6)


    up7=Conv2DTranspose(n_filters*4,kernel_size,strides=(2,2),padding='same',kernel_initializer='he_normal')
    up7=concatenate([up7,c3])
    up7=Dropout(dropout_up)(up7)
    c7=conv_batch(up7,n_filters*4,kernel_size,batch_norm,activation_fct)

    up8=Conv2DTranspose(n_filters*2,kernel_size,strides=(2,2),padding='same',kernel_initializer='he_normal')
    up8=concatenate([up8,c2])
    up8=Dropout(dropout_up)(up8)
    c8=conv_batch(up8,n_filters*2,kernel_size,batch_norm,activation_fct)

    up9=Conv2DTranspose(n_filters,kernel_size,strides=(2,2),padding='same',kernel_initializer='he_normal')
    up9=concatenate([up9,c1])
    up9=Dropout(dropout_up)(up9)
    c9=conv_batch(up9,n_filters,kernel_size,batch_norm,activation_fct)

    #final classification

    output=Conv2D(1,(1,1),final_activation)(c9)

    #building model
    model=Model(inputs=input,outputs=output)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

    return model
