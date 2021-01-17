# import tensorflow as tf
# import glob
# from tqdm import tqdm
# from pydicom import dcmread
# from PIL import Image
# import numpy as np
# from tensorflow_addons.image import rotate
#
#
# def forward_project(image):
#     image = pad_images(image)
#     angle = 0.0
#     d_angle = tf.cast(tf.constant(np.pi), tf.float32) / 454
#
#     sino_list = list()
#     for j in range(454):
#         rot_img = rotate(image, angle)
#         sino_column = tf.math.reduce_sum(rot_img, 0)
#         sino_list.append(sino_column)
#         angle += d_angle
#
#     sinogram = tf.stack(sino_list, axis=1)
#     return sinogram
#
#
# def pad_images(image):
#     last_column = image[:, -1]
#     first_column = image[:, 0]
#
#     wide_list = list()
#     for i in range(271):
#         if i < 135:
#             wide_list.append(first_column)
#         elif i > 135:
#             wide_list.append(last_column)
#         else:
#             for j in range(image.shape[0]):
#                 wide_list.append(image[:, j])
#
#     wide_image = tf.stack(wide_list, axis=1)
#     tall_list = list()
#     first_row = wide_image[0, :]
#     last_row = wide_image[-1, :]
#     for i in range(271):
#         if i < 135:
#             tall_list.append(first_row)
#         elif i > 135:
#             tall_list.append(last_row)
#         else:
#             for j in range(image.shape[0]):
#                 tall_list.append(wide_image[j, :])
#
#     big_image = tf.stack(tall_list, axis=0)
#
#     return big_image
#
#
# save_path_sinos = r"D:\BME548Project\Dataset\s_data\big\\"
#
#
# # Load DCOM Images and Save sinos:
# loop = 0
# set_num = 0
# folder_prev = ""
# for file in tqdm(glob.glob(r"D:\BME548Project\Dataset\*\*\*"
#                            r"\*.dcm",
#                            recursive=True)):
#     folder = file.split('\\')
#     if folder[-2] != folder_prev:
#         set_num += 1
#         loop = 0
#     if 500 < loop < 750:
#         dicom = dcmread(file)
#         image_array = dicom.pixel_array
#         image_array = tf.constant(image_array, dtype=tf.float32)
#         sino = forward_project(image_array)
#         arr = sino.numpy()
#         img = Image.fromarray(arr)
#         img = img.resize((454, 454))
#         save_path = save_path_sinos + 'L{0:06}'.format(set_num) + \
#                                        '{0:04}.tiff'.format(loop)
#         img.save(save_path)
#     loop += 1
#     folder_prev = folder[-2]