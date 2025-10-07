import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, Dropout, MaxPooling2D

from keras.models import load_model
model = load_model("model.10-0.04.keras")


mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

HEIGHT, WIDTH = x_train[0].shape
NCLASSES = tf.size(tf.unique(y_train).y)
print("Image height x width is", HEIGHT, "x", WIDTH)
tf.print("There are", NCLASSES, "classes")

BUFFER_SIZE = 5000
BATCH_SIZE = 100

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.expand_dims(image, -1)
    return image, label


def load_dataset(training=True):
    """Loads MNIST dataset into a tf.data.Dataset"""
    (x_train, y_train), (x_test, y_test) = mnist
    x = x_train if training else x_test
    y = y_train if training else y_test
    # One-hot encode the classes
    y = tf.keras.utils.to_categorical(y, NCLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(scale).batch(BATCH_SIZE)
    if training:
        dataset = dataset.shuffle(BUFFER_SIZE).repeat()
    return dataset

validation_data = load_dataset(training=False)

score = model.evaluate(validation_data, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


to_predict = np.array([x_test[32]])
output = model.predict(to_predict)
print('The digit is:', np.argmax(output))
pixels = to_predict[0].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
