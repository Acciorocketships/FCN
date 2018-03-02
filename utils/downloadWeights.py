import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import *
from keras.applications.vgg16 import *
from keras.applications.resnet50 import *
import keras.backend as K
import tensorflow as tf
from .resnethelpers import *



def Vgg16_weights(input_shape=(224,224,3),weights_path=None,model=None):

    if model is not None and weights_path is None:
        classes = int(model.outputs[0]._shape[-1])
        weights_path = 'weights/vgg16' + str((input_shape[0],input_shape[1],classes)).replace(" ","").replace(",","-") + ".h5"

    if not os.path.isfile(weights_path):

        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        include_top = (input_shape==(224,224,3))
        vgg16 = VGG16(input_shape=input_shape,include_top=include_top,weights='imagenet')
        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name=='fc1':
                weights[0] = np.reshape(weights[0], (7,7,512,4096))
            elif layer.name=='fc2':
                weights[0] = np.reshape(weights[0], (1,1,4096,4096))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        
    return weights_path



def Resnet50_weights(input_shape=(224,224,3),weights_path=None,model=None):

    if model is not None and weights_path is None:
        classes = int(model.outputs[0]._shape[-1])
        weights_path = 'weights/res50' + str((input_shape[0],input_shape[1],classes)).replace(" ","").replace(",","-") + ".h5"

    if not os.path.isfile(weights_path):

        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        include_top = (input_shape==(224,224,3))
        resnet50 = ResNet50(input_shape=input_shape,include_top=include_top,weights='imagenet')
        for layer in resnet50.layers:
            weights = layer.get_weights()
            if layer.name=='output':
                weights[0] = np.reshape(weights[0], (1,1,2048,classes))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        
    return weights_path



def Vgg19_weights(input_shape=(224,224,3),weights_path=None,model=None):

    if model is not None and weights_path is None:
        classes = int(model.outputs[0]._shape[-1])
        weights_path = 'weights/vgg19' + str((input_shape[0],input_shape[1],classes)).replace(" ","").replace(",","-") + ".h5"

    if not os.path.isfile(weights_path):

        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        include_top = (input_shape==(224,224,3))
        vgg16 = VGG19(input_shape=input_shape,include_top=include_top,weights='imagenet')
        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name=='fc1':
                weights[0] = np.reshape(weights[0], (7,7,512,4096))
            elif layer.name=='fc2':
                weights[0] = np.reshape(weights[0], (1,1,4096,4096))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        
    return weights_path
