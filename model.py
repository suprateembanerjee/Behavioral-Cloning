import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from time import time

lines = []

t1=time()

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:

	steering_center = float(line[3])
	correction = 0.2
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	image_center = mpimg.imread("C:\\Users\\Dolphin48\\Jupyter Notebooks\\CarND-Behavioral-Cloning-P3-master\\data\\"+str(line[0]))
	image_left = mpimg.imread("C:\\Users\\Dolphin48\\Jupyter Notebooks\\CarND-Behavioral-Cloning-P3-master\\data\\"+str(line[1][1:]))
	image_right = mpimg.imread("C:\\Users\\Dolphin48\\Jupyter Notebooks\\CarND-Behavioral-Cloning-P3-master\\data\\"+str(line[2][1:]))
	
	images.extend((image_center, image_left, image_right))
	measurements.extend((steering_center, steering_left, steering_right))

X_train = np.array(images)
y_train = np.array(measurements)

# Preprocess: Augment using flipping

X_train_flipped = []
for image in X_train:
	X_train_flipped.append(np.fliplr(image))

X_train_flipped = np.array(X_train_flipped)
y_train_flipped = np.multiply(y_train , -1.0)

X_train = np.append(X_train, X_train_flipped, axis = 0)
y_train = np.append(y_train, y_train_flipped)

""" CROPPING HAS BEEN INCORPORATED IN THE NETWORK
X_train_bottom = []
for image in X_train:
	X_train_bottom.append(image[int(image.shape[0]/2.5):,:])


X_train = np.array(X_train_bottom)

"""

#X_train, y_train = shuffle(X_train, y_train)

#X_train = (X_train - X_train.mean()) / X_train.std()

print(X_train.shape)
print(y_train.shape)

print("Time taken to preprocess: {:.2f}s".format(time() - t1))


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Cropping2D, LayerNormalization, Input, Lambda, Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import RootMeanSquaredError


def inception_v3():

	from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

	# INCEPTION_V3
	global X_train, y_train
	X_train = preprocess_input(X_train)
	inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
	inception.trainable = False
	print(inception.summary())

	driving_input = Input(shape=(96,320,3))
	resized_input = Lambda(lambda image: tf.image.resize(image,(299,299)))(driving_input)
	inp = inception(resized_input)

	x = GlobalAveragePooling2D()(inp)

	x = Dense(512, activation = 'relu')(x)
	x = Dense(256, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(128, activation = 'relu')(x)
	x = Dense(64, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	result = Dense(1, activation = 'linear')(x)

	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
	model = Model(inputs = driving_input, outputs = result)
	t2 = time()
	model.compile(optimizer=optimizer, loss=loss)

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_inc.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 10)

	model.fit(x=X_train, y=y_train, shuffle=True, validation_split=0.2, epochs=100, 
		batch_size=32, verbose=1, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_inc.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')

def nasnetLarge():

	from tensorflow.keras.applications import NASNetLarge
	from tensorflow.keras.applications.nasnet import preprocess_input

	# NASNET
	global X_train, y_train
	X_train = preprocess_nasnet(X_train)
	nasnet = NASNetLarge(weights='imagenet', include_top=False, input_shape = (331,331,3))
	nasnet.trainable = False
	print(nasnet.summary())
	nasnet_input = Input(shape=(96,320,3))
	resized_input = Lambda(lambda image: tf.image.resize(image, (331,331)))(nasnet_input)
	inp = nasnet(resized_input)
	x = GlobalAveragePooling2D()(inp)
	x = Dense(512, activation = 'relu')(x)
	x = Dense(256, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(128, activation = 'relu')(x)
	x = Dense(64, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	predictions = Dense(1, activation='linear')(x)

	model = Model(inputs=nasnet_input, outputs=predictions)
	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")

	t2 = time()
	model.compile(optimizer=optimizer, loss=loss)

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_nasnet.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 20)

	model.fit(X_train, y_train, shuffle=True, validation_split=0.2, epochs=100, 
		batch_size=32, verbose=1, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_nasnet.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')


def custom():

	# CUSTOM
	global X_train, y_train

	model = Sequential()
	model.add(Conv2D(input_shape=(96, 320, 3), filters=32, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))

	model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))

	model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1))
	model.add(Activation('linear'))

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_custom.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 10)

	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
	t2 = time()
	model.compile(loss = loss, optimizer = optimizer)
	model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,
				epochs = 100, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_custom.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')

def nvidia():

	from tensorflow.keras.layers.experimental.preprocessing import Normalization

	# NVIDIA
	global X_train, y_train

	inputs = Input(shape=(160,320,3))
	cropped = Cropping2D(cropping=((64, 0), (0, 0)))(inputs)
	resized_input = Lambda(lambda image: tf.image.resize(image, (66,200)))(cropped)
	normalize_layer = LayerNormalization(axis=1)(resized_input)
	conv1 = Conv2D(filters=24, kernel_size=5, strides=(2,2), activation='relu')(normalize_layer)
	conv2 = Conv2D(filters=36, kernel_size=5, strides=(2,2), activation='relu')(conv1)
	conv3 = Conv2D(filters=48, kernel_size=5, strides=(2,2), activation='relu')(conv2)
	conv4 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv3)
	conv5 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv4)
	flatten = Flatten()(conv5)
	dense1 = Dense(100,activation='relu')(flatten)
	dense2 = Dense(50,activation='relu')(dense1)
	dense3 = Dense(10,activation='relu')(dense2)
	out = Dense(1, activation='linear')(dense3)

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_nvidia.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 10)

	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
	t2 = time()
	model = Model(inputs=inputs, outputs=out)
	model.compile(loss = loss, optimizer = optimizer)
	history = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,
		epochs = 100, callbacks=[checkpoint, stopper])

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Huber Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training set', 'Validation set'], loc = 'upper right')
	plt.savefig('loss.png')
	plt.show()

	model.load_weights('./ckpts/model_nvidia.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')





# Run models
nvidia()