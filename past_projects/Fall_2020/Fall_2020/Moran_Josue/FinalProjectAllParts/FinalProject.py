#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:28:05 2020

@author: josue_nataren
"""

import matplotlib.pyplot as plt
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import os, os.path
import cv2
import tensorflow as tf
from keras.constraints import max_norm
from keras.utils import plot_model
#from sklearn.model_selection import StratifiedKFold

#filename = get_testdata_files("/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/1-01.dcm")[0]
#ds = pydicom.dcmread("/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/1-45.dcm")
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

# 512x512

#print((ds.pixel_array.shape))
#<matplotlib.image.AxesImage object at ...>    
    
    


def noisy(image):
    #print(image.shape)
    row,col = image.shape
    s_vs_p = 0.5
    amount = 0.03
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out




PathFilesA = "/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Acases/"
PathFilesB = "/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Bcases/"
PathFilesE = "/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Ecases/"
PathFilesG = "/Users/josue_nataren/Downloads/FINAL_PROJECT_BME548/Lung-PET-CT-Dx/Gcases/"

ClassificationLabel = ["A","B","E","G"]
#ClassificationLabel = [0,1,2,3]

#This is for A cases
totalFiles = 0
totalDir = 0
names = []
count = 0
for base, dirs, files in os.walk(PathFilesA):
    names.append(base)
    for directories in dirs:
        totalDir += 1
        #for other in files:
            #print("this is the folder:"+base+"/"+directories+"/"+other)
    for Files in files:
        totalFiles += 1

print('Total number of files in A',totalFiles)
print('Total Number of directories in A',totalDir)
print('Total in A:',(totalDir + totalFiles))

names = names[2:]
print(len(names))

allfilesA = []

for i in range(len(names)):
    for base, dirs, files in os.walk(names[i]):
        #names.append(base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
            #print(base+"/"+Files)
            dirPath = base+"/"+Files
            # if dirPath not in allfilesA: 
            #     #res.append(i)
            allfilesA.append(dirPath)
print("Number of A cases: "+str(len(allfilesA)))
allfilesA = list(dict.fromkeys(allfilesA))
print("New Number of A cases: "+str(len(allfilesA)))
#End of A cases





#This is for B cases
totalFiles = 0
totalDir = 0
names = []
count = 0
for base, dirs, files in os.walk(PathFilesB):
    names.append(base)
    for directories in dirs:
        totalDir += 1
        #for other in files:
            #print("this is the folder:"+base+"/"+directories+"/"+other)
    for Files in files:
        totalFiles += 1

print('Total number of files in B',totalFiles)
print('Total Number of directories in B',totalDir)
print('Total in B:',(totalDir + totalFiles))

names = names[2:]
print(len(names))

allfilesB = []

for i in range(len(names)):
    for base, dirs, files in os.walk(names[i]):
        #names.append(base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
            #print(base+"/"+Files)
            dirPath = base+"/"+Files
            # if dirPath not in allfilesA: 
            #     #res.append(i)
            allfilesB.append(dirPath)
print("Number of B cases: "+str(len(allfilesB)))
allfilesB = list(dict.fromkeys(allfilesB))
print("New Number of B cases: "+str(len(allfilesB)))
#End of B cases




#This is for E cases
totalFiles = 0
totalDir = 0
names = []
count = 0
for base, dirs, files in os.walk(PathFilesE):
    names.append(base)
    for directories in dirs:
        totalDir += 1
        #for other in files:
            #print("this is the folder:"+base+"/"+directories+"/"+other)
    for Files in files:
        totalFiles += 1

print('Total number of files in E',totalFiles)
print('Total Number of directories in E',totalDir)
print('Total in E:',(totalDir + totalFiles))

names = names[2:]
print(len(names))

allfilesE = []

for i in range(len(names)):
    for base, dirs, files in os.walk(names[i]):
        #names.append(base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
            #print(base+"/"+Files)
            dirPath = base+"/"+Files
            # if dirPath not in allfilesA: 
            #     #res.append(i)
            allfilesE.append(dirPath)
print("Number of E cases: "+str(len(allfilesE)))
allfilesE = list(dict.fromkeys(allfilesE))
print("New Number of E cases: "+str(len(allfilesE)))
#End of E cases





#This is for G cases
totalFiles = 0
totalDir = 0
names = []
count = 0
for base, dirs, files in os.walk(PathFilesG):
    names.append(base)
    for directories in dirs:
        totalDir += 1
        #for other in files:
            #print("this is the folder:"+base+"/"+directories+"/"+other)
    for Files in files:
        totalFiles += 1

print('Total number of files in G',totalFiles)
print('Total Number of directories in G',totalDir)
print('Total in G:',(totalDir + totalFiles))

names = names[2:]
print(len(names))

allfilesG = []

for i in range(len(names)):
    for base, dirs, files in os.walk(names[i]):
        #names.append(base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
            #print(base+"/"+Files)
            dirPath = base+"/"+Files
            # if dirPath not in allfilesA: 
            #     #res.append(i)
            allfilesG.append(dirPath)
print("Number of G cases: "+str(len(allfilesG)))
allfilesG = list(dict.fromkeys(allfilesG))
print("New Number of G cases: "+str(len(allfilesG)))
#End of G cases
    

# print("Loading a test image")  
# ds = pydicom.dcmread(allfilesA[100])
# ds = ds.pixel_array
# print(ds.shape)
# ds = cv2.resize(ds, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
# print(ds.shape)

# dsnoisy = noisy(ds)
# dsnoisy = np.asarray(dsnoisy)

# fig, axs = plt.subplots(2)
# axs[0].imshow(ds, cmap=plt.cm.bone)
# axs[1].imshow(dsnoisy, cmap=plt.cm.bone)


#plt.imshow(ds, cmap=plt.cm.bone)

# ds = ds.flatten()
# print(ds.shape)

numImg = 15000
sizeOfImg = 150


EvectorImages = np.zeros((len(allfilesE), sizeOfImg, sizeOfImg, 1))
EvectorOfClass = np.zeros((len(allfilesE), 4))
#ClassificationLabel = ["A","B","E","G"]     [0,0,1,0]
#ClassificationLabel = [0,1,2,3]
#for i in range(100):
for i in range(len(allfilesE)):
    dslocal = pydicom.dcmread(allfilesE[i])
    dslocal = dslocal.pixel_array
    #dslocal = np.resize(dslocal,(150,150))
    dslocal = cv2.resize(dslocal, dsize=(sizeOfImg, sizeOfImg), interpolation=cv2.INTER_CUBIC)
    #dslocal = dslocal.flatten()
    #EvectorImages.append(dslocal)
    if len(dslocal.shape) > 2:
        #...
        if dslocal.shape[2] >=1:
            r, g, b = dslocal[:,:,0], dslocal[:,:,1], dslocal[:,:,2]
            dslocal = 0.2989 * r + 0.5870 * g + 0.1140 * b
            EvectorImages[i,:,:,0] = dslocal
    else:
        EvectorImages[i,:,:,0] = dslocal
    EvectorOfClass[i,:] = [0,0,1,0]
    #EvectorOfClass.append(2)
print("Done with the E images")
#EvectorImages = np.asarray(EvectorImages)
print("Size: "+str(EvectorImages.shape)+" with "+str(EvectorOfClass.shape))


GvectorImages = np.zeros((numImg, sizeOfImg, sizeOfImg, 1))
GvectorOfClass = np.zeros((numImg, 4))
#ClassificationLabel = ["A","B","E","G"]     [0,0,0,1]
#ClassificationLabel = [0,1,2,3]
for j in range(numImg):
#for j in range(len(allfilesG)):
    dslocal = pydicom.dcmread(allfilesG[j])
    dslocal = dslocal.pixel_array
    #dslocal = np.resize(dslocal,(150,150))
    dslocal = cv2.resize(dslocal, dsize=(sizeOfImg, sizeOfImg), interpolation=cv2.INTER_CUBIC)
    #dslocal = dslocal.flatten()
    #GvectorImages.append(dslocal)
    if len(dslocal.shape) > 2:
        #...
        if dslocal.shape[2] >=1:
            r, g, b = dslocal[:,:,0], dslocal[:,:,1], dslocal[:,:,2]
            dslocal = 0.2989 * r + 0.5870 * g + 0.1140 * b
            GvectorImages[j,:,:,0] = dslocal
    else:
        GvectorImages[j,:,:,0] = dslocal
    GvectorOfClass[j,:] = [0,0,0,1]
    #GvectorOfClass.append(2)
print("Done with the G images")
#GvectorImages = np.asarray(GvectorImages)
print("Size: "+str(GvectorImages.shape)+" with "+str(GvectorOfClass.shape))


AvectorImages = np.zeros((numImg, sizeOfImg, sizeOfImg, 1))
AvectorOfClass = np.zeros((numImg, 4))
#ClassificationLabel = ["A","B","E","G"]     [1,0,0,0]
#ClassificationLabel = [0,1,2,3]
for k in range(numImg):
#for k in range(len(allfilesA)):
    dslocal = pydicom.dcmread(allfilesA[k])
    dslocal = dslocal.pixel_array
    #dslocal = np.resize(dslocal,(150,150))
    dslocal = cv2.resize(dslocal, dsize=(sizeOfImg, sizeOfImg), interpolation=cv2.INTER_CUBIC)
    #dslocal = dslocal.flatten()
    #AvectorImages.append(dslocal)
    if len(dslocal.shape) > 2:
        #...
        if dslocal.shape[2] >=1:
            r, g, b = dslocal[:,:,0], dslocal[:,:,1], dslocal[:,:,2]
            dslocal = 0.2989 * r + 0.5870 * g + 0.1140 * b
            AvectorImages[k,:,:,0] = dslocal
    else:
        AvectorImages[k,:,:,0] = dslocal
    AvectorOfClass[k,:] = [1,0,0,0]
    #AvectorOfClass.append(2)
print("Done with the A images")
#AvectorImages = np.asarray(AvectorImages)
print("Size: "+str(AvectorImages.shape)+" with "+str(AvectorOfClass.shape))


BvectorImages = np.zeros((numImg, sizeOfImg, sizeOfImg, 1))
BvectorOfClass = np.zeros((numImg, 4))
#ClassificationLabel = ["A","B","E","G"]     [0,1,0,0]
#ClassificationLabel = [0,1,2,3]
for m in range(numImg):
#for m in range(len(allfilesB)):
    dslocal = pydicom.dcmread(allfilesB[m])
    dslocal = dslocal.pixel_array
    #dslocal = np.resize(dslocal,(150,150))
    dslocal = cv2.resize(dslocal, dsize=(sizeOfImg, sizeOfImg), interpolation=cv2.INTER_CUBIC)
    #dslocal = dslocal.flatten()
    #BvectorImages.append(dslocal)
    if len(dslocal.shape) > 2:
        #...
        if dslocal.shape[2] >=1:
            r, g, b = dslocal[:,:,0], dslocal[:,:,1], dslocal[:,:,2]
            dslocal = 0.2989 * r + 0.5870 * g + 0.1140 * b
            BvectorImages[m,:,:,0] = dslocal
    else:
        BvectorImages[m,:,:,0] = dslocal
    BvectorOfClass[m,:] = [0,1,0,0]
    #BvectorOfClass.append(2)
print("Done with the B images")
#BvectorImages = np.asarray(BvectorImages)
print("Size: "+str(BvectorImages.shape)+" with "+str(BvectorOfClass.shape))


x_data = EvectorImages
y_data = EvectorOfClass
x_data = np.append(x_data, BvectorImages, axis=0)
y_data = np.append(y_data, BvectorOfClass, axis=0)
x_data = np.append(x_data, AvectorImages, axis=0)
y_data = np.append(y_data, AvectorOfClass, axis=0)
x_data = np.append(x_data, GvectorImages, axis=0)
y_data = np.append(y_data, GvectorOfClass, axis=0)


indicesX = np.arange(y_data.shape[0])
np.random.shuffle(indicesX)

trainNoise, validateNoise, trainNormal, validateNormal, testFinalValues = np.split(indicesX, [int(indicesX.shape[0]*0.35), int(indicesX.shape[0]*0.5), int(indicesX.shape[0]*0.85), int(indicesX.shape[0]*0.95)])

x_trainNoise = x_data[trainNoise,:,:,:]
y_trainNoise = y_data[trainNoise,:]
x_validateNoise = x_data[validateNoise,:,:,:]
y_validateNoise = y_data[validateNoise,:]
x_train = x_data[trainNormal,:,:,:]
y_train = y_data[trainNormal,:]
x_validate = x_data[validateNormal,:,:,:]
y_validate = y_data[validateNormal,:]
x_testFinal = x_data[testFinalValues,:,:,:]
y_testFinal= y_data[testFinalValues,:]


print("The size of x train for noise is: "+str(x_trainNoise.shape)+"; and y train for noise: "+str(y_trainNoise.shape))
print("The size of x validate for noise is: "+str(x_validateNoise.shape)+"; and y validate for noise: "+str(y_validateNoise.shape))
print("The size of x train normal is: "+str(x_train.shape)+"; and y train normal: "+str(y_train.shape))
print("The size of x validate normal is: "+str(x_validate.shape)+"; and y validate normal: "+str(y_validate.shape))

#TWO DIFFERENT MODELS: 1 for the denoiser and 1 for classification
#Then merge them together so that the input of the final model takes a noisy image and still classifies it.


#This is the denoiser model:
pure = np.copy(x_trainNoise)
pure_test = np.copy(x_validateNoise)
noisy_input = np.copy(pure)
noisy_input_test = np.copy(pure_test)
for i in range(noisy_input.shape[0]):
    localThing = np.copy(noisy_input[i,:,:,0])
    noisy_input[i,:,:,0] = noisy(localThing)
for i in range(noisy_input_test.shape[0]):
    localThing = np.copy(noisy_input_test[i,:,:,0])
    noisy_input_test[i,:,:,0] = noisy(localThing)
    

print("Loading a test image")  
fig, axs = plt.subplots(2)
axs[0].imshow(pure[1000,:,:,0], cmap=plt.cm.bone)
axs[1].imshow(noisy_input[1000,:,:,0], cmap=plt.cm.bone)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((150,150, 1)))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2), padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2)))
model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2)))
model.add(tf.keras.layers.UpSampling2D((2,2)))
model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2)))
model.add(tf.keras.layers.UpSampling2D((2,2)))
model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), activation='sigmoid'))
model.summary()
# Compile and fit data
metrics = ['acc']
model.compile(optimizer='adam', metrics = metrics, loss='binary_crossentropy')
history = model.fit(noisy_input, pure, epochs=5, batch_size=64, validation_data=(noisy_input_test, pure_test))

#Plotting
#fig, axs = plt.subplots(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy vs Epochs for Denoiser only')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epochs for Denoiser only')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#This is the classification model
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Input((150,150, 1)))
model2.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="relu"))
model2.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu"))
# model2.add(tf.keras.layers.MaxPool2D(2))
model2.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu"))
model2.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu"))
# model2.add(tf.keras.layers.MaxPool2D(2))
model2.add(tf.keras.layers.Dense(32, activation="relu"))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(4, activation="softmax"))

model2.summary()

loss_fn_name = "categorical_crossentropy"
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model2.compile(loss=loss_fn_name, metrics = metrics, optimizer=opt)

# training
history1 = model2.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_validate, y_validate))
#test_perf = model2.evaluate(x_val, y_val, batch_size=64)


#Plotting
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('Accuracy vs Epochs for Classifier only')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Loss vs Epochs for Classifier only')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




modelfinal = tf.keras.models.Sequential()
modelfinal.add(tf.keras.layers.Input((150,150, 1)))
modelfinal.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
modelfinal.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
modelfinal.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation='relu', kernel_constraint=max_norm(2), kernel_initializer='he_uniform'))
modelfinal.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2), padding='same'))
modelfinal.add(tf.keras.layers.UpSampling2D((2,2)))
modelfinal.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2)))
modelfinal.add(tf.keras.layers.UpSampling2D((2,2)))
modelfinal.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(3,3), activation='relu', kernel_constraint=max_norm(2)))
modelfinal.add(tf.keras.layers.UpSampling2D((2,2)))
modelfinal.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), activation='sigmoid'))

modelfinal.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="relu"))
modelfinal.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu"))
modelfinal.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu"))
modelfinal.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu"))
modelfinal.add(tf.keras.layers.Dense(32, activation="relu"))
modelfinal.add(tf.keras.layers.Flatten())
modelfinal.add(tf.keras.layers.Dense(4, activation="softmax"))

modelfinal.layers[1].set_weights(model.layers[1].get_weights())
modelfinal.layers[2].set_weights(model.layers[2].get_weights())
modelfinal.layers[3].set_weights(model.layers[3].get_weights())
modelfinal.layers[4].set_weights(model.layers[4].get_weights())
modelfinal.layers[5].set_weights(model.layers[5].get_weights())
modelfinal.layers[6].set_weights(model.layers[6].get_weights())
modelfinal.layers[7].set_weights(model.layers[7].get_weights())
modelfinal.layers[8].set_weights(model.layers[8].get_weights())
modelfinal.layers[9].set_weights(model.layers[9].get_weights())
#modelfinal.layers[10].set_weights(model.layers[10].get_weights())
#modelfinal.layers[5].set_weights(model.layers[5].get_weights())

modelfinal.layers[11].set_weights(model2.layers[1].get_weights())
modelfinal.layers[12].set_weights(model2.layers[2].get_weights())
modelfinal.layers[13].set_weights(model2.layers[3].get_weights())
modelfinal.layers[14].set_weights(model2.layers[4].get_weights())
modelfinal.layers[15].set_weights(model2.layers[5].get_weights())
modelfinal.layers[16].set_weights(model2.layers[6].get_weights())
#modelfinal.layers[12].set_weights(model2.layers[7].get_weights())

modelfinal.summary()

modelfinal.compile(loss=loss_fn_name, metrics = metrics, optimizer=opt)

# training
pure = np.copy(x_train)
pure_test = np.copy(x_testFinal)
noisy_input = np.copy(pure)
noisy_input_test = np.copy(pure_test)
for i in range(noisy_input.shape[0]):
    localThing = np.copy(noisy_input[i,:,:,0])
    noisy_input[i,:,:,0] = noisy(localThing)
for i in range(noisy_input_test.shape[0]):
    localThing = np.copy(noisy_input_test[i,:,:,0])
    noisy_input_test[i,:,:,0] = noisy(localThing)


#By this point is already trained basically
history2 = modelfinal.fit(noisy_input, y_train, epochs=5, batch_size=64, validation_data=(noisy_input_test, y_testFinal))


# samples = noisy_input_test[:9]
# denoised_images = model.predict(samples)



#print(history.history[])

#Plotting
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('Accuracy vs Epochs for Final CNN')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Loss vs Epochs for Final CNN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plot_model(model, to_file='model_denoiser.png')
plot_model(model2, to_file='model_classifier.png')
plot_model(modelfinal, to_file='model_complete.png')
