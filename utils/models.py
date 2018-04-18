import numpy as np
import os
import sys
import h5py
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from .basics import *
from .resnetHelpers import *



def Vgg16(input_shape=(224,224,3), classes=1, regularization=0.):
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(regularization))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(regularization))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(regularization))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(regularization))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(regularization))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(regularization))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2), name='fc1', kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(regularization))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    return model



def Vgg19(input_shape=(224,224,3), classes=1, regularization=0.):
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2), name='fc1', kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(regularization))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    return model



def Resnet50(input_shape=(224,224,3), classes=1, regularization=0., batch_momentum=0.9):
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(regularization))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=regularization, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=regularization, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=regularization, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=regularization, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=regularization, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=regularization, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=regularization, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=regularization, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(regularization))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(regularization))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    
    return model



def UNet(input_shape=(224,224,3), classes=1, weights_path=None, regularization=0.):
    
    img_input = Input(shape=input_shape)

    # 224 x 224 x 3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(regularization))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(regularization))(x)
    x1 = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    # 112 x 112 x 64

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(regularization))(x)
    x2 = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # 56 x 56 x 128

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(regularization))(x)
    x3 = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # 28 x 28 x 256

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(regularization))(x)
    x4 = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # 14 x 14 x 512

    # Block 5
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(regularization))(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(regularization))(x)
    x5 = Dropout(0.2)(x)

    # 14 x 14 x 1024

    # Deconv Block 4T
    x = Conv2DTranspose(512, (2,2), strides=(2,2), activation='relu', padding='same', name="block4T_deconv1", kernel_regularizer=l2(regularization))(x5)
    x = Concatenate(axis=2)([x,x4])
    x = Conv2D(512, (2,2), activation='relu', padding='same', name="block4T_conv1", kernel_regularizer=l2(regularization))(x)
    x = Conv2D(256, (2,2), activation='relu', padding='same', name="block4T_conv2", kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.2)(x)

    # 28 x 28 x 512

    # Deconv Block 3T
    x = Conv2DTranspose(256, (2,2), strides=(2,2), activation='relu', padding='same', name="block3T_deconv1", kernel_regularizer=l2(regularization))(x)
    x = Concatenate(axis=2)([x,x3])
    x = Conv2D(256, (2,2), activation='relu', padding='same', name="block3T_conv1", kernel_regularizer=l2(regularization))(x)
    x = Conv2D(128, (2,2), activation='relu', padding='same', name="block3T_conv2", kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.2)(x)

    # 56 x 56 x 128

    # Deconv Block 2T
    x = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu', padding='same', name="block2T_deconv1", kernel_regularizer=l2(regularization))(x)
    x = Concatenate(axis=2)([x,x2])
    x = Conv2D(128, (2,2), activation='relu', padding='same', name="block2T_conv1", kernel_regularizer=l2(regularization))(x)
    x = Conv2D(64, (2,2), activation='relu', padding='same', name="block2T_conv2", kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.2)(x)

    # 112 x 112 x 64

    # Deconv Block 1T
    x = Conv2DTranspose(64, (2,2), strides=(2,2), activation='relu', padding='same', name="block1T_deconv1", kernel_regularizer=l2(regularization))(x)
    x = Concatenate(axis=2)([x,x1])
    x = Conv2D(64, (2,2), activation='relu', padding='same', name="block1T_conv1", kernel_regularizer=l2(regularization))(x)
    x = Conv2D(64, (2,2), activation='relu', padding='same', name="block1T_conv2", kernel_regularizer=l2(regularization))(x)
    x = Dropout(0.2)(x)

    # Classifying layer
    x = Conv2DTranspose(classes, (1, 1), activation='linear', padding='same', name='classes', kernel_regularizer=l2(regularization), kernel_initializer='he_normal')(x)
    # changed activation

    model = Model(img_input, x)

    return model
