import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np
from tensorflow import keras
from PIL import Image
import json


class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_name, img_path, batch_size=32, dim=(208, 176), n_channels=1, n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        with open(file_name) as f:
            data = f.read()
        self.data = json.loads(data)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.img_path = img_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        data_list_temp = [self.data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_list_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, data in enumerate(data_list_temp):
            # Store sample
            img = np.array(Image.open(self.img_path + data['image_id']))

            X[i,] = img[..., np.newaxis]

            # Store class
            y[i] = data['label']

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

file_name = 'D:/桌面文件2/出国准备/Duke/BME590OpticsImaging/final_project/Alzheimer_s Dataset/'
training_generator = DataGenerator('train_notes.json', file_name)

file_name = 'D:/桌面文件2/出国准备/Duke/BME590OpticsImaging/final_project/Alzheimer_s Dataset/'
test_generator = DataGenerator('test_notes.json', file_name)


class MRIPhysicalLayer(keras.layers.Layer):
    def __init__(self, filters, in_channels, kernel_size, strides, activation):
        self.filters = filters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.strides = strides
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = tf.nn.relu if activation == 'relu' else tf.nn.sigmoid
        super(MRIPhysicalLayer, self).__init__()

    def build(self, input_shape, training=None):
        # expect the input_shape to be (batch_size, height, width, filters_previous)
        batch_size, height, width, filters_previous = input_shape
        self.kernel = self.add_weight(name='kernel',
                                      shape=[self.kernel_size[0], self.kernel_size[1], self.in_channels, self.filters],
                                      initializer='normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=[self.filters],
                                    initializer='zero',
                                    trainable=True)
        super(MRIPhysicalLayer, self).build(input_shape)

    def call(self, x, training=None):
        spatial_kernel2 = tf.reverse(self.kernel, axis=[0])
        spatial_kernel3 = tf.reverse(self.kernel, axis=[1])
        spatial_kernel4 = tf.reverse(self.kernel, axis=[0, 1])
        spatial_kernel12 = tf.concat([self.kernel, spatial_kernel2], axis=0)
        spatial_kernel34 = tf.concat([spatial_kernel3, spatial_kernel4], axis=0)
        spatial_kernel = tf.concat([spatial_kernel12, spatial_kernel34], axis=1)

        train_x = self.activation(tf.nn.conv2d(x, spatial_kernel, self.strides, "SAME") + self.bias)

        # test_x = self.activation(tf.nn.conv2d(x, spatial_kernel, self.strides, "SAME")+self.bias)

        return train_x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[1] / self.strides), int(input_shape[2] / self.strides), self.filters)

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalMaxPooling2D, UpSampling2D, \
    Dense, MaxPooling2D, Dropout, Flatten, Lambda, GlobalAveragePooling2D, Concatenate, Average, Activation, Add,ZeroPadding2D
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dropout
from os import path

kfjNet = Sequential([
        #MRIPhysicalLayer(1, in_channels=1, kernel_size=[13, 11], activation=None, strides=1),
        Conv2D(5, kernel_size=5, activation="relu", strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(5, kernel_size=5, activation="relu", strides=1),
        Conv2D(5, kernel_size=5, activation="relu", strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(5, kernel_size=5, activation="relu", strides=1),
        Conv2D(5, kernel_size=5, activation="relu", strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax"),
    ])



x_in = Input(shape=(208, 176, 1))
y = kfjNet(x_in)
model4 = Model(inputs=x_in, outputs=y)

optimizer = Adam(lr=0.001)
# callbacks = [
#             ModelCheckpoint(
#                 path.join("weights.best.h5"),
#                 save_weights_only=True,
#                 save_best_only=True,
#                 monitor='val_acc',
#                 mode='auto'
#             )
#         ]
callbacks = [
            ModelCheckpoint(
                path.join("weights.full.best.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor='val_acc',
                mode='auto'
            )
        ]
model4.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", "categorical_crossentropy"],
    )
model4.summary()

model4.fit_generator(
        training_generator,
        validation_data=test_generator,
        validation_freq=1,
        epochs=30,
        verbose=1,
        callbacks=callbacks
    )

model4.save_weights("model_full_mask.h5")
