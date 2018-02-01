import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import *
from keras.applications.vgg16 import *
from keras.applications.resnet50 import *
import keras.backend as K
import tensorflow as tf

from .get_weights_path import *
from .resnet_helpers import *

def transfer_FCN_Vgg16(input_shape=(224,224,3),weights_path=None,classes=1):

    if weights_path is None:
        weights_path = 'weights/vgg16' + str(input_shape).replace(" ","").replace(",","-") + ".h5"

    if not os.path.isfile(weights_path):

        img_input = Input(shape=input_shape)
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
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Convolutional layers transfered from fully-connected layers
        x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
        x = Conv2D(classes, (1, 1), activation='linear', name='output')(x)
        #x = Reshape((7,7))(x)

        # Create model
        model = Model(img_input, x)

        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        vgg16 = VGG16()
        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name=='fc1':
                weights[0] = np.reshape(weights[0], (7,7,512,4096))
            elif layer.name=='fc2':
                weights[0] = np.reshape(weights[0], (1,1,4096,4096))
            elif layer.name=='output':
                weights[0] = np.reshape(weights[0], (1,1,4096,classes))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        
    return weights_path

 

def transfer_FCN_ResNet50(input_shape=(224,224,3),weights_path=None,classes=1):

    if weights_path is None:
        weights_path = 'weights/res50' + str(input_shape).replace(" ","").replace(",","-") + ".h5"

    if not os.path.isfile(weights_path):

        img_input = Input(shape=input_shape)
        bn_axis = 3

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

        x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

        x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

        x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
        x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
        x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)

        x = Conv2D(classes, (1, 1), activation='linear', name='output')(x)

        # Create model
        model = Model(img_input, x)

        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        resnet50 = ResNet50()
        for layer in resnet50.layers:
            weights = layer.get_weights()
            if layer.name=='output':
                weights[0] = np.reshape(weights[0], (1,1,2048,classes))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        
        return weights_path


if __name__ == '__main__':
    modelname = None
    if len(sys.argv) < 2 or sys.argv[1] not in {'Vgg16', 'ResNet50'}:
        modelname = 'Vgg16'
    else:
        modelname = sys.argv[1]
    func = globals()['transfer_FCN_%s'%modelname]
    func()
