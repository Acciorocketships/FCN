from fcn import *

classifier = FCN('vgg16',7,(224,224,3),regularization=0.2)

classifier.train('train/img','train/label',epochs=50,val_split=0.1,zoom=0.4,rotation=180,xshift=0.3,yshift=0.8,shear=0.2,colorshift=0.2)