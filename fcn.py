import os
from random import randrange
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from utils.models import *
from utils.downloadWeights import transfer_FCN_Vgg16, transfer_FCN_ResNet50
from utils.SegDataGenerator import *
from utils.lossFunction import *
from utils.metrics import *
from skimage.transform import resize


class FCN:

	def __init__(self,model,classes,input_shape=(224,224,3),optimizer=None,learning_rate=0.001,regularization=0.):
		self.input_shape = input_shape
		self.classes = classes
		if isinstance(model,Model):
			self.weights_path = 'custom_model.h5'
			self.model = model
		elif 'vgg16' in model:
			self.weights_path = transfer_FCN_Vgg16(input_shape=input_shape,classes=classes)
			self.model = FCN_Vgg16_32s(input_shape=input_shape,weights_path=self.weights_path,classes=classes,regularization=regularization)
		elif 'res50' in model:
			self.weights_path = transfer_FCN_ResNet50(input_shape=input_shape,classes=classes)
			self.model = FCN_ResNet50_32s(input_shape=input_shape,weights_path=self.weights_path,classes=classes)
		loss_fn=softmax_sparse_crossentropy_ignoring_last_label
		if optimizer is None:
			optimizer = SGD(lr=learning_rate, momentum=0.9)
		metrics=[sparse_accuracy_ignoring_last_label]
		self.model.compile(loss=loss_fn,
					  optimizer=optimizer,
					  metrics=metrics)


	def datagen(self,data_dir,label_dir,file_txt,batch_size=8,
				zoom=0,rotation=0,xshift=0,yshift=0,shear=0,xflip=False,yflip=False,colorshift=0):
		gen = SegDataGenerator(zoom_range=zoom,
                                 zoom_maintain_shape=True,
                                 crop_mode='random',
                                 crop_size=self.input_shape[0:2],
                                 rotation_range=rotation,
                                 shear_range=shear,
                                 width_shift_range=xshift,
                                 height_shift_range=yshift,
                                 horizontal_flip=xflip,
                                 vertical_flip=yflip,
                                 channel_shift_range=colorshift,
                                 fill_mode='constant')
		return gen.flow_from_directory(data_dir=data_dir,label_dir=label_dir, #save_to_dir='weights/preprocessed',
									   file_path=file_txt,classes=self.classes,batch_size=batch_size)


	def dir_images(self,directory):
		return list(map(lambda x: os.path.splitext(x)[0], \
					filter(lambda x: os.path.splitext(x)[1]=='.png' or os.path.splitext(x)[1]=='.jpg', \
					os.listdir(directory))))


	def train_generators(self,data_dir,label_dir,val_split,batch_size=8,
			  			 zoom=0,rotation=0,xshift=0,yshift=0,shear=0,xflip=False,yflip=False,
			  			 colorshift=0):
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
							   xshift=xshift,yshift=yshift,shear=shear,xflip=xflip,yflip=yflip, \
							   colorshift=colorshift)
		if len(labels) > 0:
			gen['val'] = self.datagen(data_dir,label_dir,'val.txt', \
							   batch_size=batch_size,zoom=zoom,rotation=rotation, \
							   xshift=xshift,yshift=yshift,shear=shear,xflip=xflip,yflip=yflip, \
							   colorshift=colorshift)
		return gen


	def train(self,data_dir,label_dir,val_split=0.,batch_size=8,epochs=5,
			  zoom=0,rotation=0,xshift=0,yshift=0,shear=0,xflip=False,yflip=False,
			  colorshift=0):
		gen = self.train_generators(data_dir,label_dir,val_split,batch_size=batch_size,
								    zoom=zoom,rotation=rotation, xshift=xshift,yshift=yshift, \
								    shear=shear,xflip=xflip,yflip=yflip,colorshift=colorshift)
		if 'val' in gen:
			history = self.model.fit_generator(generator=gen['train'],validation_data=gen['val'],epochs=epochs)
		else:
			history = self.model.fit_generator(generator=gen['train'],epochs=epochs)
		self.model.save_weights(self.weights_path)
		return history
		

	# also handle if img is a list of images or a directory
	def predict(self,img):
		img = resize(img,self.input_shape[0:2])
		return self.model.predict(img)