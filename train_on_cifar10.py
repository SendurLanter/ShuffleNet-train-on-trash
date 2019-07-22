import sys, os,cv2
from keras.callbacks import TensorBoard
import keras
import numpy as np
from model import MobileNet
from keras.preprocessing import image
batch_size = 64
num_classes = 2
epochs = 20

x_train = np.empty((77,224,224,3),dtype="float32")
x_test=list()
y_train=list()
y_test=list()
def load():
	tra_i=0
	tes_i=0
	datas = os.listdir('./data')
	print(datas)
	for e in datas:
		img = cv2.imread('./data/'+e)
		if e[0] == 'p':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			x_train[tra_i] = img
			y_train.append([0])
			tra_i+=1
		if e[0] == 'a':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			x_train[tra_i] = img
			y_train.append([1])
			tra_i+=1
	return (x_train,np.array(y_train)) , (x_test,np.array(y_test))

(x_train, y_train), (x_test, y_test) = load()
print(x_train.shape)
print(y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train /= 255
print(x_train.shape)
print(y_train.shape)
img_input = keras.layers.Input(shape=(224, 224, 3))
model = MobileNet(input_tensor=img_input, classes=num_classes)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer = keras.optimizers.rmsprop(lr=0.0004, decay=5e-4),metrics=['accuracy'])
model.fit(x_train, y_train, validation_split =0.2,batch_size = batch_size, epochs = epochs, verbose = 1,shuffle=True)