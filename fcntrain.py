from fcn import *

classifier = FCN(model='vgg16',classes=8,input_shape=(224,224,3))

classifier.train('train/img','train/label',epochs=5,val_split=0.1,zoom=0.1,rotation=90,shear=0.2,colorshift=0.2,tensorboard='weights/tensorboard/vgg16(224-224-8)')