import numpy as np
import cv2
from skimage.measure import label
from imgstream import Stream
from fcn import FCN


class Tracker:

	def __init__(self,classes=None):
		self.classes = classes
		# Feature extractor and matcher
		self.features = cv2.xfeatures2d.SIFT_create()
		index_params = dict(algorithm=0,trees=5)
		search_params = dict(checks=50)
		self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
		# Values to store
		self.kp = None
		self.desc = None


	def update(self,img,mask):
		numclasses = mask.shape[2]-1
		if self.classes is None:
			self.classes = [i for i in range(1,numclasses+1)]
		mask = np.argmax(mask,axis=2)
		self.kp = [None]*numclasses
		self.desc = [None]*numclasses
		for i in range(numclasses):
			labeled, numregions = label((mask==i+1),return_num=True)
			self.kp[i] = [[]]*numregions
			self.desc[i] = [[]]*numregions
			for j in range(numregions):
				cvmask = (labeled==j+1)
				cvmask.dtype='uint8'
				self.kp[i][j], self.desc[i][j] = self.features.detectAndCompute(img,cvmask)


	def objects(self,img):
		obj = {cls: [] for cls in self.classes}
		kp, desc = self.features.detectAndCompute(img,None)
		for i in range(len(self.kp)):
			for j in range(len(self.kp[i])):
				matches = self.matcher.knnMatch(self.desc[i][j],desc,k=1)
				keypoints = []
				for [match] in matches:
					if match.distance < 100:
						keypoints.append(kp[match.trainIdx])
				if len(keypoints) > 0:
					obj[self.classes[i]].append(Object(keypoints,self.classes[i]))
		return obj


class Object:

	def __init__(self,keypoints=[],classification=None):
		self.classification = classification
		self.mu = np.array([[0.],[0.]])
		self.cov = np.array([[0.,0.],[0.,0.]])
		self.compute(keypoints)

	def compute(self,keypoints):
		for kp in keypoints:
			self.mu += np.array([[kp.pt[0]],[kp.pt[1]]])
		self.mu /= len(keypoints)
		for kp in keypoints:
			pt1 = np.array([[kp.pt[0]+kp.size/2],[kp.pt[1]]])
			pt2 = np.array([[kp.pt[0]-kp.size/2],[kp.pt[1]]])
			pt3 = np.array([[kp.pt[0]],[kp.pt[1]+kp.size/2]])
			pt4 = np.array([[kp.pt[0]],[kp.pt[1]-kp.size/2]])
			self.cov += (pt1-self.mu) @ (pt1-self.mu).T
			self.cov += (pt2-self.mu) @ (pt2-self.mu).T
			self.cov += (pt3-self.mu) @ (pt3-self.mu).T
			self.cov += (pt4-self.mu) @ (pt4-self.mu).T
		self.cov /= 4*len(keypoints) - 1

	def eval(self,pt):
		# 0.1 is approximately the boundary of the pulse
		pt = np.array(pt)
		if len(pt.shape) == 1:
			pt = np.expand_dims(pt,axis=1)
		return np.exp(-0.5 * (pt-self.mu).T @ np.linalg.inv(self.cov) @ (pt-self.mu)) / (2*np.pi * np.sqrt(np.linalg.det(self.cov)))
		
	def eig(self):
		# eigvals[i] corresponds with eigvecs[:,i]
		eigvals, eigvecs = np.linalg.eig(self.cov)
		return (eigvals, eigvecs)

	def size(self):
		# size([[1,0],[0,1]]) == 1
		return np.sqrt(np.linalg.det(self.cov))

	def rotation(self):
		# In radians. If the longer axis is horizontal, rotation is 0
		_, eigvecs = self.eig()
		mainaxis = eigvecs[:,0]
		return (np.arctan(mainaxis[0]/mainaxis[1]) if mainaxis[1]!=0 else np.pi)

	def eccentricity(self):
		# e=0 is a circle. e=1 is a line.
		_, eigvals = self.eig()
		return np.sqrt(1 - (eigvals[1]/eigvals[0])**2)





if __name__ == '__main__':
	tracker = Tracker()
	stream = Stream(mode='cam',src='facetest')
	classifier = FCN(model='Vgg16',classes=8,input_shape=(224,224,3))
	for i,img in enumerate(stream):
		if i % 5 == 0:
			mask = classifier.predict(img)
			tracker.update(img,mask)
		else:
			objects = tracker.objects(img)