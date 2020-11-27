import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

	image_center = mpimg.imread("./data/"+str(line[0]))
	image_left = mpimg.imread("./data/"+str(line[1][1:]))
	image_right = mpimg.imread("./data/"+str(line[2][1:]))
	
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

plt.imsave('flipped.png',X_train_flipped[0])
plt.imsave('unflipped.png',X_train[0])

X_train = np.append(X_train, X_train_flipped, axis = 0)
y_train = np.append(y_train, y_train_flipped)

print(X_train.shape)
print(y_train.shape)

print("Time taken to preprocess: {:.2f}s".format(time() - t1))



from tensorflow.keras.layers import Cropping2D, LayerNormalization, Input, Lambda, Dense, Flatten, Conv2D, Activation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay


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