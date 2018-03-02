from fcn import *

model = 'vgg16'
shape = (224,224,3)

classifier = FCN(model=model,classes=8,input_shape=shape,loss_size_weight=2,loss_weights={0:0.4,1:0.7,3:2.0})

classifier.train('train/img','train/label',epochs=300,initial_epoch=0,val_split=None,
				 zoom=0.1,rotation=90,shear=0.2,colorshift=0.2,autosave=True,
				 tensorboard='weights/tensorboard/' + model + '-' + str(shape[0]) + '-' + str(shape[1]) + '-8')