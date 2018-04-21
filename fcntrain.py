from fcn import *

model = 'vgg16fcn'
shape = (224,224,3)

classifier = FCN(model=model,classes=8,input_shape=shape,loss_size_weight=1)

classifier.train('train/img','train/label',epochs=300,initial_epoch=0,val_split=None,
				 zoom=0.2,rotation=180,shear=0.3,colorshift=0.3,autosave=True,
				 tensorboard='weights/tensorboard/' + model + '-' + str(shape[0]) + '-' + str(shape[1]) + '-8')