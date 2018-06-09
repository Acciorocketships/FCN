from fcn import *

model = 'Vgg16'
shape = (224,224,3)
classes = 1

classifier = FCN(model=model,classes=classes,input_shape=shape,regression=True)

classifier.train('train/depthimg','train/depthlabel',epochs=300,initial_epoch=0,val_split=0.1,
				 zoom=0.0,rotation=180,shear=0.3,colorshift=0.3,autosave=True,learning_rate=4e-8,
				 tensorboard='weights/tensorboard/' + model + '-' + str(shape[0]) + '-' + str(shape[1]) + '-' + str(classes))