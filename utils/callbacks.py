from keras.callbacks import Callback
from imgstream import Stream

class Visualization(Callback):
    def __init__(self, path, classifier):
        self.stream = Stream(src=path,mode='img')
        self.classifier = classifier

    def on_batch_end(self, epoch, logs={}):
        try:
            for img in self.stream:
                mask = self.classifier.predict(img)
                masked = Stream.mask(mask,img,alpha=0.5)
                Stream.show(masked,'Training Visualization',pause=False,shape=(480,640))
        except:
            pass