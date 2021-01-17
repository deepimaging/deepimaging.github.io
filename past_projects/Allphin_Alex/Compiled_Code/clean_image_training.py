import os
import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import Image


from unet_model import my_unet


def normalize_image(x):
    x = x / np.max(x)
    return x


if __name__ == "__main__":
    initial_e = 0
    epochs = 5
    for i in range(4):
        x_data = list()
        unfiltered = list()
        y_data = list()
        for file in tqdm(
                glob.glob(r'D:\BME548Project\Dataset\x_data\small\*.png')):
            lower_range = np.random.randint(0, 1000)
            upper_range = lower_range + 50
            if lower_range < int(file[-8:-4]) < upper_range:
                im = Image.open(file)
                img = np.asarray(im)
                unfiltered.append(img)
                img = normalize_image(img)
                x_data.append(np.expand_dims(img, axis=2))

                filey = os.path.split(file)[1]
                pathy = r'D:\BME548Project\Dataset\y_data\small'
                full_pathy = os.path.join(pathy, filey)
                im = Image.open(full_pathy)
                img = np.asarray(im)
                y_data.append(np.expand_dims(img, axis=2))

        x_data = np.array(x_data)
        unfiltered = np.array(unfiltered)
        y_data = np.array(y_data)
        model_checkpoint = ModelCheckpoint('basic_model.h5',
                                           save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
        if os.path.isfile('basic_model.h5'):
            model = load_model('basic_model.h5')
        else:
            model = my_unet()
            optimizer = tf.keras.optimizers.Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
                'accuracy'])

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            test_size=0.3)
        print("Starting Training")
        history = model.fit(x_train, y_train, initial_epoch=initial_e,
                            epochs=epochs,
                            batch_size=2,
                            validation_data=(x_test, y_test), callbacks=[
                             model_checkpoint, tensorboard_callback])

        img_number = 50

        orig = Image.fromarray(unfiltered[img_number], mode='F')
        orig.save(f'original.tiff')
        labe = Image.fromarray(y_data[img_number, :, :, 0], mode='I')
        labe.save('label.tiff')
        pred_image = np.expand_dims(x_data[img_number], axis=0)
        pred_image = np.expand_dims(pred_image, axis=3)
        y_pred = model.predict(pred_image)[0, :, :, 0]
        tmp = y_pred
        mask_predict = Image.fromarray(tmp, mode='F')
        mask_predict.save(f'predicted{epochs}.tiff')

        initial_e += 5
        epochs += 5