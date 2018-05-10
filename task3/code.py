# import libraries
import keras
import tensorflow as tf
import scipy.io
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential
import matplotlib.pyplot as plt
#% matplotlib inline
plt.rcParams["figure.dpi"] = 200
np.set_printoptions(precision=3, suppress=True)
K.set_image_data_format('channels_first')
K.image_data_format()
batch_size = 128
num_classes = 10
epochs = 20
# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (3, img_rows, img_cols)

# load dataset
train = scipy.io.loadmat('train_32x32.mat')
test = scipy.io.loadmat('test_32x32.mat')
X_train = train['X']
y_train = train['y']
X_test = test['X']
y_test = test['y']

# preprocessing
X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np.ndarray.flatten(y_train)
y_test = np.ndarray.flatten(y_test)
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0
num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convolutional NN
num_classes = 10
cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(X_train, y_train,
                      batch_size=128, epochs=20, verbose=1, validation_split=.1)
score = cnn.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Convolutional NN with Batch Normalization
model_bn = Sequential([
    Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
model_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_bn = model_bn.fit(X_train, y_train, batch_size=128,
                    epochs=10, verbose=1, validation_split=.1)
bn_score = model_bn.evaluate(X_test, y_test, verbose=0)
print('Test loss:', bn_score[0])
print('Test accuracy:', bn_score[1])

## Convolutional NN with Batch Normalization-2

num_classes = 10
cnn32_bn = Sequential()
cnn32_bn.add(Conv2D(64, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Conv2D(64, (3, 3)))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Flatten())
cnn32_bn.add(Dense(64, activation='relu'))
cnn32_bn.add(Dense(num_classes, activation='softmax'))

cnn32_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_32_bn = cnn32_bn.fit(X_train, y_train,
                                 batch_size=128, epochs=10, verbose=1, validation_split=.1)
