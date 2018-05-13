from fcn import *

model = 'Vgg16Multiscale'
shape = (480,640,3)
classes = 2

classifier = FCN(model=model,classes=classes,input_shape=shape,loss_size_weight=2,loss_weights={1:2},class_swaps={1:0,3:1,7:0,6:0,5:0,4:0,2:0})

classifier.train('train/droneimg','train/dronelabel',epochs=300,initial_epoch=0,val_split=0.1,
				 zoom=0.2,rotation=180,shear=0.3,colorshift=0.3,autosave=True,learning_rate=1e-3,
				 tensorboard='weights/tensorboard/' + model + '-' + str(shape[0]) + '-' + str(shape[1]) + '-' + str(classes))