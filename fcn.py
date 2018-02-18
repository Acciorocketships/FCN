import os
from random import randrange
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from utils.models import *
from utils.downloadweights import transfer_FCN_Vgg16, transfer_FCN_ResNet50
from utils.segdatagen import *
from utils.metrics import *
from skimage.transform import resize


class FCN:

	def __init__(self,model='vgg16',classes=1,input_shape=(224,224,3),optimizer=None,loss=None,accuracy=None,learning_rate=0.001,regularization=0.):
		self.input_shape = input_shape
		self.classes = classes
		if isinstance(model,Model):
			self.weights_path = 'custom_model.h5'
			self.model = model
		elif 'vgg16' in model:
			self.weights_path = transfer_FCN_Vgg16(input_shape=input_shape,classes=classes)
			self.model = AtrousFCN_Vgg16_16s(input_shape=input_shape,weights_path=self.weights_path,classes=classes,regularization=regularization)
		elif 'res50' in model:
			self.weights_path = transfer_FCN_ResNet50(input_shape=input_shape,classes=classes)
			self.model = AtrousFCN_ResNet50_16s(input_shape=input_shape,weights_path=self.weights_path,classes=classes,regularization=regularization)
		if loss is None:
			loss = softmax_sparse_crossentropy_ignoring_last_label
		if optimizer is None:
			optimizer = SGD(lr=learning_rate, momentum=0.9)
		if accuracy is None:
			accuracy = [sparse_accuracy_ignoring_last_label]
		else:
			accuracy = [accuracy]
		self.model.compile(loss=loss,
					  optimizer=optimizer,
					  metrics=accuracy)
		self.gen = SegDataGenerator()


	def datagen(self,data_dir,label_dir,file_txt,batch_size=8,
				zoom=0.,rotation=0,shear=0.,xflip=False,yflip=False,
				normalization=False,sample_normalization=False,
				colorshift=0.,savedir=None):
		self.gen = SegDataGenerator(zoom_range=zoom,
                                 zoom_maintain_shape=True,
                                 crop_mode='random',
                                 crop_size=self.input_shape[0:2],
                                 rotation_range=rotation,
                                 shear_range=shear,
                                 horizontal_flip=xflip,
                                 vertical_flip=yflip,
                                 featurewise_center=normalization,
                                 featurewise_std_normalization=normalization,
                                 samplewise_center=sample_normalization,
                                 samplewise_std_normalization=sample_normalization,
                                 channel_shift_range=colorshift,
                                 fill_mode='constant')
		if normalization:
			from imgstream import Stream
			stream = Stream(mode='img',src='data_dir')
			imgs = np.array(list(stream))
			self.get.fit(imgs)
		return self.gen.flow_from_directory(data_dir=data_dir,label_dir=label_dir,save_to_dir=savedir,
									   file_path=file_txt,classes=self.classes,batch_size=batch_size)


	def dir_images(self,directory):
		return list(map(lambda x: os.path.splitext(x)[0], \
					filter(lambda x: os.path.splitext(x)[1]=='.png' or os.path.splitext(x)[1]=='.jpg', \
					os.listdir(directory))))


	def train_generators(self,data_dir,label_dir,val_split,batch_size=8,
			  			 zoom=0,rotation=0,shear=0,xflip=False,yflip=False,
			  			 normalization=False,sample_normalization=False,
			  			 colorshift=0,savedir=None):

		if val_split is not None:
			data = self.dir_images(data_dir)
			labels = []
			for i in range(int(val_split*len(data))):
				idx = randrange(0,len(data))
				labels.append(data[idx])
				del data[idx]
			with open('train.txt','w') as f:
				for name in data:
					f.write(name)
					f.write('\n')
			if len(labels) > 0:
				with open('val.txt','w') as f:
					for name in labels:
						f.write(name)
						f.write('\n')
		gen = {}
		gen['train'] = self.datagen(data_dir,label_dir,'train.txt', \
							   batch_size=batch_size,zoom=zoom,rotation=rotation, \
							   shear=shear,xflip=xflip,yflip=yflip, \
							   normalization=normalization,sample_normalization=sample_normalization, \
							   colorshift=colorshift,savedir=savedir)
		if len(labels) > 0:
			gen['val'] = self.datagen(data_dir,label_dir,'val.txt', \
							   batch_size=batch_size,zoom=zoom,rotation=rotation, \
							   shear=shear,xflip=xflip,yflip=yflip, \
							   normalization=normalization,sample_normalization=sample_normalization, \
							   colorshift=colorshift,savedir=savedir)
		return gen

	# Inputs:
		# val_split: ratio of validation data to total data (float). If val_split is None, then it uses filenames in train.txt and val.txt.
		# https://keras.io/preprocessing/image/
		# savedir: directory to save preprocessed images and labels to (string)
		# tensorboard: path/filename to save tensorboard logs (string)
		# callbacks: callback, or list of callbacks (https://keras.io/callbacks/)
	def train(self,data_dir,label_dir,val_split=0.,batch_size=8,epochs=5,
			  zoom=0,rotation=0,shear=0,xflip=False,yflip=False,
			  normalization=False,sample_normalization=False,
			  colorshift=0,savedir=None,tensorboard=None,callbacks=[]):
		if not isinstance(callbacks,list):
			callbacks = [callbacks]
		if tensorboard is not None:
			callbacks.append(TensorBoard(log_dir=tensorboard, histogram_freq=0, write_graph=True, write_images=True))
			try:
				import subprocess
				subprocess.Popen(["tensorboard","--logdir",tensorboard])
				import webbrowser
				webbrowser.open("http://localhost:6006",new=2)
			except:
				pass
		gen = self.train_generators(data_dir,label_dir,val_split,batch_size=batch_size,zoom=zoom,rotation=rotation,shear=shear, \
									xflip=xflip,yflip=yflip,normalization=normalization,sample_normalization=sample_normalization,
									colorshift=colorshift,savedir=savedir)
		try:
			if 'val' in gen:
				history = self.model.fit_generator(generator=gen['train'],validation_data=gen['val'],epochs=epochs,callbacks=callbacks)
			else:
				history = self.model.fit_generator(generator=gen['train'],epochs=epochs,callbacks=callbacks)
		except KeyboardInterrupt:
			print('Stopping Training...')
			self.model.save_weights(self.weights_path)
			return None
		self.model.save_weights(self.weights_path)
		return history
		

	# Inputs:
		# image as np array (height,width,channels)
		# batch of images as np array or list (samples,height,width,channels)
		# path to image or video
		# path to directory of images and/or videos
	# Output:
		# mask as np array (height,width,classes)
		# batch of masks if batch of inputs is given (samples,height,width,classes)
	def predict(self,img):
		if isinstance(img,list) or (isinstance(img,np.ndarray) and len(img.shape)==4):
			output = []
			for image in img:
				output.append(self.predict(image))
			return np.array(output)
		elif isinstance(img,str):
			from imgstream import Stream
			output = []
			stream = Stream(mode='img',src=img)
			for image in stream:
				output.append(self.predict(image))
			stream = Stream(mode='vid',src=img)
			for image in stream:
				output.append(self.predict(image))
			return np.array(output)
		img = np.array(img)
		if self.gen.featurewise_center or self.gen.samplewise_center:
			img = self.gen.standardize(img)
		shape = img.shape[:2]
		img = resize(img,self.input_shape,preserve_range=True,mode='constant')
		img = np.expand_dims(img,axis=0)
		mask = self.model.predict(img)[0]
		mask = softmax(mask,axis=2)
		mask = resize(mask,shape,preserve_range=True,mode='constant')
		return mask