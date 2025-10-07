import os
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

# Load EMNIST dataset from .mat file
def load_emnist():
    mat = loadmat("emnist-byclass.mat")
    x_train = mat['dataset'][0][0][0][0][0][0].reshape(-1, 28, 28)
    y_train = mat['dataset'][0][0][0][0][0][1].flatten()
    x_test = mat['dataset'][0][0][1][0][0][0].reshape(-1, 28, 28)
    y_test = mat['dataset'][0][0][1][0][0][1].flatten()
    return (x_train, y_train), (x_test, y_test)

# Load dataset
(x_train, y_train), (x_test, y_test) = load_emnist()
NCLASSES = len(np.unique(y_train))
print(f"Number of classes: {NCLASSES}")

# Image properties
HEIGHT, WIDTH = 28, 28
BATCH_SIZE = 128
BUFFER_SIZE = 5000

def scale(image, label):
    image = tf.expand_dims(tf.cast(image, tf.float32) / 255.0, axis=-1)
    return image, tf.keras.utils.to_categorical(label, NCLASSES)

# Prepare datasets
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(scale).batch(BATCH_SIZE)

def get_model():
    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(HEIGHT, WIDTH, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(728, activation='relu'),
        Dropout(0.3),
        Dense(400, activation='relu'),
        Dropout(0.5),
        Dense(NCLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training parameters
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100

t1 = time.perf_counter()
model = get_model()

# Define Callbacks
checkpoint_callback = ModelCheckpoint(filepath="emnist_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')
tensorboard_callback = TensorBoard(log_dir="logs")

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=2,
    callbacks=[checkpoint_callback, tensorboard_callback]
)
t2 = time.perf_counter()
print(f"Training took {t2 - t1:.4f} seconds.")

# Save final model
model.save("emnist_model.keras")
