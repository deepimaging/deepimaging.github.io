#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:23:28 2020

@author: hsm
"""

import numpy as np
import keras
import os
from fnmatch import fnmatch
import tensorflow as tf
import scipy.io as sio
from keras import backend as K
from matplotlib import pyplot as plt
 
 
from sklearn.model_selection import KFold # tool for getting random folds in K-fold cross validation
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.models import load_model
import h5py
import nibabel as nib



from tensorflow.python.client import device_lib 


tf.config.experimental.list_physical_devices('GPU') 

# print(device_lib.list_local_devices())
# config = tf.ConfigProto( device_count = {'GPU': 1} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)




def dice_metric(y_true, y_pred):

    threshold = 0.5

    # mask = y_pred > threshold
    # mask_pred = tf.cast(mask, dtype=tf.float32)
    # y_pred = tf.multiply(y_pred, mask)
    # mask = y_true > threshold
    # mask = tf.cast(mask, dtype=tf.float32)
    # y_true = tf.multiply(y_true, mask)
    # y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    # y_true = tf.cast(y_true > threshold, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    # new haodong
    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def custom_loss(y_true, y_pred):
    #return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)



image = nib.load('/home/alex/Stroke/data_full_110920.nii')
data = image.get_fdata()
data_y_1 = data[:,:,:,3]
data_y_1 = np.rot90(data_y_1,axes=(0,2))
data_y_1 = np.rot90(data_y_1,axes=(1,2))
nonzero1_train = np.transpose(np.nonzero(data_y_1[0:25*16,:,:]))
nonzero1_train = list(((nonzero1_train[:,0])))
seen_train = set()
uniq_train = []
for x in nonzero1_train:
    if x not in seen_train:
        uniq_train.append(x)
        seen_train.add(x)

data_cropped_train = data[:,:,uniq_train,:]
data_x_train = data_cropped_train[:,:,:,0]
data_x_train = np.rot90(data_x_train,axes=(0,2))
data_x_train = np.rot90(data_x_train,axes=(1,2))
data_x_train = np.reshape(data_x_train,[len(data_x_train),448,512,1])

data_y_train = data_cropped_train[:,:,:,3]
data_y_train = np.rot90(data_y_train,axes=(0,2))
data_y_train = np.rot90(data_y_train,axes=(1,2))
data_y_train = np.reshape(data_y_train,[len(data_y_train),448,512,1])




nonzero1_test= np.transpose(np.nonzero(data_y_1[25*16:25*24,:,:]))
nonzero1_test = list(((nonzero1_test[:,0])))
seen_test = set()
uniq_test = []
for x in nonzero1_test:
    if x not in seen_test:
        uniq_test.append(x)
        seen_test.add(x)

data_cropped_test = data[:,:,uniq_test,:]
data_x_test = data_cropped_test[:,:,:,0]
data_x_test = np.rot90(data_x_test,axes=(0,2))
data_x_test = np.rot90(data_x_test,axes=(1,2))
data_x_test = np.reshape(data_x_test,[len(data_x_test),448,512,1])

data_y_test = data_cropped_test[:,:,:,3]
data_y_test = np.rot90(data_y_test,axes=(0,2))
data_y_test = np.rot90(data_y_test,axes=(1,2))
data_y_test = np.reshape(data_y_test,[len(data_y_test),448,512,1])



x_train = np.copy(data_x_train)
y_train = np.copy(data_y_train)

x_val = np.copy(data_x_test)
y_val= np.copy(data_y_test)





def cnn(fliter_num,kernel_size):
    input_layer = keras.layers.Input(shape=(448, 512, 1))
    conv1a = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(input_layer)
    conv1b = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv1a)
    pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
    conv2a = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool1)
    conv2b = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv2a)
    pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
    conv3a = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool2)
    conv3b = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv3a)

    dconv3a = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(conv3b)
    dconv3b = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(dconv3a)
    unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
    cat2 = keras.layers.concatenate([conv2b, unpool2])
    dconv2a = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(cat2)
    dconv2b = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(dconv2a)
    unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
    cat1 = keras.layers.concatenate([conv1b, unpool1])
    dconv1a = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(cat1)
    dconv1b = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(dconv1a)

    output = keras.layers.Conv2D(filters=1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(dconv1b)

    model = keras.models.Model(inputs=input_layer, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_metric])
    model.summary()
    history = model.fit(x_train, y_train, epochs=200, batch_size = 16, validation_data=(x_val, y_val))



    train_loss = history.history['loss']
    np.save('/home/alex/Stroke/results/train_loss_f_k1.npy',train_loss)
    train_acc = history.history['dice_metric']
    np.save('/home/alex/Stroke/results/train_acc_f_k1.npy',train_acc)
    val_loss = history.history['val_loss']
    np.save('/home/alex/Stroke/results/val_loss_f_k1.npy',val_loss)
    val_acc = history.history['val_dice_metric']
    np.save('/home/alex/Stroke/results/val_acc_f_k1.npy',val_acc)

    model.save('/home/alex/Stroke/results/model_f_1.h5')





cnn(64,7)