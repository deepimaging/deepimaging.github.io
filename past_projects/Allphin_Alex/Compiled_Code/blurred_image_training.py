import os
import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt


from unet_model import my_unet



initial_e = 0
epochs = 5
x_data = list()
unfiltered = list()
y_data = list()
i = 0
for file in tqdm(
        glob.glob(r'D:\BME548Project\Dataset\x_data\small\x_upload\*.png')):
    file_name = os.path.split(file)[1]
    im = Image.open(file)
    img = np.asarray(im)
    unfiltered.append(img)

    blur_path = r'D:\BME548Project\Dataset\blurred'
    file_blur = os.path.join(blur_path, file_name[:-4]+'.tiff')
    im_blur = Image.open(file_blur)
    img_blur = np.asarray(im_blur)
    img_blur = img_blur * 1000 / np.max(img_blur)
    x_data.append(np.expand_dims(img_blur, axis=2))

    pathy = r'D:\BME548Project\Dataset\y_data\small\y_upload'
    full_pathy = os.path.join(pathy, file_name)
    im = Image.open(full_pathy)
    img = np.asarray(im)
    y_data.append(np.expand_dims(img, axis=2))

    i += 1
    if i > 1000:
        break
x_data = np.array(x_data)
unfiltered = np.array(unfiltered)
y_data = np.array(y_data)



# RERUN FROM HERE DOWN TO RESUME AND CONTINUE TRAINING:
model_checkpoint = ModelCheckpoint('blurred_model.h5',
                                   save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
if os.path.isfile('blurred_model.h5'):
    model = load_model('blurred_model.h5')
else:
    model = my_unet()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        'accuracy'])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.3)
print("Starting Training")
history = model.fit(x_train, y_train, initial_epoch=initial_e,
                    epochs=epochs,
                    batch_size=10,
                    validation_data=(x_test, y_test), callbacks=[
                     model_checkpoint, tensorboard_callback])


# THIS JUST SHOWS SOME IMAGES TO CHECK HOW WE ARE DOING:
img_number = 25

original = unfiltered[img_number]
blurred = x_data[img_number, :, :, 0]
label_ = y_data[img_number, :, :, 0]

pred_image = np.expand_dims(x_data[img_number], axis=0)
pred_image = np.expand_dims(pred_image, axis=3)
y_pred = model.predict(pred_image)[0, :, :, 0]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplot(2, 2)
ax1.imshow(original, cmap='gray')
ax1.title.set_text('Original')
ax2.imshow(blurred, cmap='gray')
ax2.title.set_text('Back Projected')
ax3.imshow(label_, cmap='gray')
ax3.title.set_text('Label')
ax4.imshow(y_pred, cmap='gray')
ax4.title.set_text('Predicted Mask')
plt.show()

initial_e += 5
epochs += 10