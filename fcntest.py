from imgstream import Stream
from fcn import *

classifier = FCN(model='vgg16',classes=8,input_shape=(480,640,3))

stream = Stream(mode='img',src='test')
for img in stream:
	mask = classifier.predict(img)
	combined = stream.mask(mask[:,:,1:4],img,alpha=0.2)
	stream.show(combined,"Output",shape=(480,640),pause=True)
	


# 0 Nothing
# 1 Person
# 2 Ground
# 3 Drone
# 4 Tree
# 5 Building
# 6 Car
# 7 Sky