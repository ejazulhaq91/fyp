import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Activation, experimental
from keras.layers import Flatten, Dense
from keras import optimizers
from testface import detecting_faces
import numpy as np

import pathlib

data_directory = 'C:\\Users\\user\\PycharmProjects\\finalProject\\faces'
data_directory = pathlib.Path(data_directory)

train_data = tf.keras.preprocessing.image_dataset_from_directory(data_directory, validation_split=0.10,
                                                                 subset='training', seed=123, image_size=(256, 256),
                                                                 batch_size=16)
validation_data = tf.keras.preprocessing.image_dataset_from_directory(data_directory, validation_split=0.10,
                                                                      subset='validation', seed=123,
                                                                      image_size=(256, 256),
                                                                      batch_size=16)

class_names = train_data.class_names
num_classes = len(class_names)
# print(class_names)
# print(num_classes)


def face_model():

    train_model = Sequential()

    train_model.add(experimental.preprocessing.Rescaling(1. / 255, input_shape=(256, 256, 3)))

    train_model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    train_model.add(MaxPool2D(pool_size=2))

    train_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    train_model.add(MaxPool2D(pool_size=2))

    train_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    train_model.add(MaxPool2D(pool_size=2))

    train_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    train_model.add(MaxPool2D(pool_size=2))

    train_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    train_model.add(MaxPool2D(pool_size=2))

    train_model.add(Dropout(0.2))

    train_model.add(Flatten())

    train_model.add(Dense(512, activation='relu'))
    train_model.add(Dense(num_classes))

    train_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    train_model.fit(train_data, epochs=10, validation_data=validation_data)

    return train_model


model = face_model()
img = detecting_faces()
img = keras.preprocessing.image.load_img(img, target_size=(256, 256))
img = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img, 0)
output = model.predict(img_array)
score = tf.nn.softmax(output[0])
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100*np.max(score)))
