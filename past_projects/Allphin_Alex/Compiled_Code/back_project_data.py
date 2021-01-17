import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tqdm import tqdm
from PIL import Image
from tensorflow_addons.image import rotate


def normalize_image(in_data):
    out_image = in_data/np.max(in_data)
    return out_image


def backproject(in_tensor, filter_tensor):
    fbp_filter = filter_tensor
    sinogram = tf.cast(in_tensor, tf.complex64)
    pi = 3.14159

    # Filter Sinogram
    projLen, numAngles = sinogram.shape
    filtSinos = list()
    for j in range(numAngles):
        projfft = tf.signal.fftshift(tf.signal.fft(sinogram[:, j]))
        filtProj = projfft * tf.cast(fbp_filter, tf.complex64)
        filtSinos.append(tf.math.real(
            tf.signal.ifft(tf.signal.ifftshift(filtProj))))
    filtSino = tf.stack(filtSinos, axis=1)

    # Backproject:
    imageLen = tf.cast(sinogram.shape[0], tf.float32)
    theta = tf.linspace(0.0, 180.0, 227)
    theta = tf.scalar_mul(pi / 180.0, theta)
    numAngles = theta.shape[0]

    recon = tf.zeros((imageLen, imageLen))
    angle = 0.0
    d_angle = tf.cast(pi, tf.float32) / tf.cast(numAngles, tf.float32)
    for n in range(numAngles):
        # Get projection
        s = filtSino[:, 2 * n]

        # Create matrix with projection along center axis ready to be rotated:
        zero_layer = tf.zeros(s.shape[0])
        stack = list()
        for k in range(s.shape[0]):
            stack.append(s)
        flat_mat = tf.stack(stack, axis=0)

        # BackProject/Rotate projection matrix for desired angle:
        rot_mat = rotate(flat_mat, -angle)
        recon = tf.math.add(rot_mat, recon)
        angle += d_angle

    # Crop out empty area of image:
    crop = recon[67:387, 67:387]
    return crop


save_path = r'D:\BME548Project\Dataset\blurred'

i = 0
for file in tqdm(glob.glob(r'D:\BME548Project\Dataset\s_data\big\*.tiff')):
    filt_arr = np.random.rand(454)
    filt_tensor = tf.constant(filt_arr)

    im = Image.open(file)
    img = np.asarray(im)
    img = normalize_image(img)

    tensor_img = tf.constant(img)

    blurred_img = backproject(tensor_img, filt_tensor).numpy()

    img_obj = Image.fromarray(blurred_img)
    file_name = os.path.split(file)[1]
    path = os.path.join(save_path, file_name)
    img_obj.save(path)
