# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

# Load data from MNIST data set
MNIST = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = MNIST.load_data()

# Reshape data
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the model
# Using Neural Network:
# Stacking 3 layers of network: Conv2D, MaxPooling2D, Flatten, and Dense
model = Sequential()

# First Layer (Conv2D)
# 28 x 28 grey scale input
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Divide pixels in image to 2x2 blocks
# Pooling: takes the dominant pixel (the one that is closest to 255) and
# shrinking the image size
model.add(MaxPooling2D(2,2))

# Flatten conversts the matrix produced by MaxPooling and shrink it into
# a 1D array to connect to the following layers
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu')) # relu stands for rectified linear unit
model.add(Dense(10, activation='softmax'))


#######
#Compile the model
#######

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#####
# Training the model
#####

history = model.fit(train_images, train_labels, epochs=10)


####
# Evaluating model
####

model.evaluate(test_images, test_labels)

