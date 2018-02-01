from fcn import *
from keras.applications.vgg16 import *

#model = VGG16(weights='imagenet',input_shape=(224,224,3))

classifier = FCN('vgg16',7,(224,224,3),regularization=0.2)

classifier.train('train/img','train/label',val_split=0.1,zoom=0.4,rotation=180,xshift=0.3,yshift=0.8,shear=0.2,colorshift=0.2)