from imgstream import Stream
from fcn import *

classifier = FCN('vgg16',7,(224,224,3),regularization=0.2)

stream = Stream(mode='img',src='train/img')
for img in stream:
	mask = classifier.predict(img)
	stream.show(img, "Input")
	stream.show(mask[:,:,0], "Person")
	stream.show(mask[:,:,1], "Ground")
	stream.show(mask[:,:,2], "Drone", pause=True)



# 0 Person
# 1 Ground
# 2 Drone
# 3 Tree
# 4 Building
# 5 Car
# 6 Sky



# DIDNT WORK (WHY??)

# from imgstream import Stream
# from fcn import FCN
# import numpy as np

# stream = Stream(mode='img',src='train/img')
# classifier = FCN('vgg16',7,(224,224,3),regularization=0.2)

# for img in stream:
# 	mask = classifier.predict(np.expand_dims(img,axis=0))
# 	stream.show(img, "Input")
# 	stream.show(mask, "Output", pause=True)