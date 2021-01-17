import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve
import skimage
from skimage import util


def random_noise(image,noise_num):

    img_noise = image
    rows, cols = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y] = 255
    return img_noise


def get_conv_ker(img, img_Noise):
    f_image = np.fft.fftshift(np.fft.fft2(img))
    f_Nimage = np.fft.fftshift(np.fft.fft2(img_Noise))
    f_conv_ker = f_Nimage/f_image 
    print(f_conv_ker.shape)

    img_ker = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(f_conv_ker)))

    return img_ker


def read_img(case_path):
    itkimage = sitk.ReadImage(case_path)  
    OR = itkimage.GetOrigin() 
    SP = itkimage.GetSpacing() 
    img_array = sitk.GetArrayFromImage(itkimage)  
    return img_array, OR, SP


def articraft(img):
    x, y = img.shape[0], img.shape[1]
    for i in range(x):
        if i % 10==0:
            for j in range(y):
                img[i, j] = -3.02e+03
    return img


def p_noise(img):
    image = np.copy(img)
    img_noise=util.random_noise(image, mode='Poisson', seed=1)
    return img_noise


def g_noise(img):
    image = np.copy(img)
    img_noise=util.random_noise(image, mode='gaussian')
    return img_noise


if __name__ == '__main__':
    case_path = './1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd'
    img_array, OR, SP = read_img(case_path)

    img = np.copy(img_array[68, :, :])
    img2 = np.copy(img_array[68, :, :])
    img3 = np.copy(img_array[68, :, :])

    img_noise = g_noise(img)

    img_p_ker = get_conv_ker(img, p_noise(img3))
    img_noise = p_noise(img3)
    print(np.std(img_noise, ddof=1))

    plt.show()

