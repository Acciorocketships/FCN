import sys, os
sys.path.append(os.path.dirname(os.path.realpath("")))
from imgstream import Stream
from fcn import *

shape = (224,224,3)
classifier = FCN(model='vgg16',classes=8,input_shape=shape)

labels = ['Nothing','Person','Ground','Drone','Tree','Building','Car','Sky']
stream = Stream(mode='img',src='test')
for i,img in enumerate(stream):
	img = Stream.resize(img,shape[:2])
	mask = classifier.predict(img)
	confidence = Stream.mask(mask[:,:,1:], img, alpha=0.2,argmax=False, labels=labels[1:])
	argmax = Stream.mask(mask[:,:,:],img, alpha=0.5, argmax=True, labels=labels)
	masklayer = Stream.mask(mask[:,:,3], alpha=0.5, labels=[labels[3]])
	#Stream.show(masklayer,"Drone",pause=False)
	Stream.show(confidence,"Confidence",pause=False,shape=(shape[:2]))
	#Stream.show(argmax,"Argmax",pause=True,shape=shape[:2])