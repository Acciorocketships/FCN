import keras.backend as K
import tensorflow as tf
from keras.objectives import *
from keras.metrics import binary_crossentropy
from tensorflow.contrib.metrics import streaming_mean_iou
import numpy as np

class LossWeights:
    values = []
    func = lambda y_true, i: 1
    @staticmethod
    def setValues(weights):
    # dict or list of manual multipliers for the relative weights between classes
    # if you want class index 3 to be 2x more important to get right, set {3:2} or [1,1,1,2,1,...]
        if isinstance(weights,dict) and len(weights)>0:
            length = max(weights.keys())
            LossWeights.values = [1]*(length+1)
            for cls in weights.keys():
                LossWeights.values[cls] = weights[cls]
        elif isinstance(weights,list) or isinstance(weights,np.ndarray):
            LossWeights.values = weights
    @staticmethod
    def setSizeWeight(weight):
    # weight = 0 means smaller classes and larger classes will have the same weight
    # if weight is int/float: coefficient c = exp(weight - weight * sizeofclass / (numpixels / numclasses) )
    # if weight is 'inv', then coefficient c = (numpixels / numclasses) / sizeofclass
        if weight == 'inv':
            LossWeights.func = lambda y_true, i: ( tf.to_float(tf.shape(y_true)[0]) / tf.to_float(tf.shape(y_true)[1]) ) / K.sum(y_true[:,i])
        elif weight != 0:
            LossWeights.func = lambda y_true, i: tf.exp( weight  -  weight * K.sum(y_true[:,i]) / ( tf.to_float(tf.shape(y_true)[0]) / tf.to_float(tf.shape(y_true)[1]) ) )
    @staticmethod
    def apply(y_true):
        if len(LossWeights.values) < int(y_true.shape[1]):
            LossWeights.values = LossWeights.values + [1]*(int(y_true.shape[1])-len(LossWeights.values))
        output = []
        for i in range(len(LossWeights.values)):
            coeff = LossWeights.values[i] * LossWeights.func(y_true,i)
            mask = coeff * y_true[:,i]
            output.append(mask)
        return tf.stack(output,axis=1)


def argmax_accuracy(y_true,y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),nb_classes)
    return K.sum(tf.to_float(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / tf.to_float(tf.shape(y_true)[0])


def testsize(y_true,y_pred):
    # Makes incompatible shapes error more informative.
    return K.sum(tf.to_float(K.equal(K.argmax(y_true,axis=-1),K.argmax(y_pred,axis=-1))))


def softmax_crossentropy_loss(y_true,y_pred):
    ### Higher weight penalizes MISSES for that class more ###
    # Hit: y_true[i]=1, log_softmax[i]=0; cross_entropy = 0, false_alarms = 0
    # Correct Rejection: y_true=0, log_softmax=-inf; cross_entropy = 0 false_alarms = 0
    # Miss: y_true[i]=1, log_softmax=-inf; cross_entropy = inf, false_alarms = 0
    # False Alarm: y_true=0, log_softmax=0; cross_entropy = 0, false_alarms = inf
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    false_alarms = K.sum((1-y_true) * (-1/(log_softmax - 0.1)), axis=1)
    false_alarms_mean = K.mean(false_alarms)
    y_true = LossWeights.apply(y_true)
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    return cross_entropy_mean # + false_alarms_mean


def regression_mean_squared_loss(y_true,y_pred):
    y_pred = tf.to_float(K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1])))
    y_true = tf.to_float(K.reshape(y_true, (-1, K.int_shape(y_pred)[-1])))
    sqdiff = K.square(y_pred - y_true)
    mse = K.mean(sqdiff)
    return mse


# X: ndarray
# axis: axis to compute values along. Default is the first non-singleton axis.
def softmax(X, axis=None):
    theta = 1.0
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

    