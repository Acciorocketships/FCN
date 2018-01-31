from fcn import *
from keras.applications.vgg16 import *

#model = VGG16(weights='imagenet',input_shape=(224,224,3))

classifier = FCN('vgg16',7,(224,224,3))

classifier.train('train/img','train/label',0.2)