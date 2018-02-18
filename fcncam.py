from imgstream import Stream
from fcn import *

classifier = FCN('vgg16',7,(224,224,3),regularization=0.)

stream = Stream(mode='cam')
for img in stream:
	mask = classifier.predict(img)
	combined = stream.mask(mask[:,:,1:4],img,alpha=0.2)
	stream.show(combined,"Output",shape=(720,1280))
	


# 0 Nothing
# 1 Person
# 2 Ground
# 3 Drone
# 4 Tree
# 5 Building
# 6 Car
# 7 Sky