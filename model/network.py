"""
This script is created for the realization of the model where we have to deploy the deep learning architecture
I will start with basic CNN architecture and then implement the one of the paper as well as implementing
LSTM or prediction.
"""


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, MaxPool3D, GlobalAveragePooling3D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K


directory = os.getcwd() + '/data'

tensor = np.load(directory+'/tensor_1000_2000.npy')
labels = np.load(directory+'/labels_1000_2000.npy')

x_train, x_test, y_train, y_test = train_test_split(tensor, labels, test_size=0.20, random_state=42)
x_train = np.reshape(x_train, (800, 1, 10, 10, 10))
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

input_shape = (1, 10, 10, 10)
batch_size = 32
epochs = 24
num_filters = 10

## MODEL
model = Sequential()
model.add(Conv3D(10, kernel_size=(3, 3, 3), activation='relu', data_format='channels_first', kernel_initializer='he_uniform', batch_input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last', padding='same')) ## da non fare
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(10, kernel_size=(1, 1, 1), activation='relu', data_format='channels_first', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_first', padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()
# Fit data to model

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
