import glob
import nibabel as nib
from tqdm import tqdm
from pydicom import dcmread
from time import sleep
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# save_path_images = r"D:\BME548Project\Dataset\x_data\\"
# save_path_labels = r"D:\BME548Project\Dataset\y_data\\"


# Load DCOM Images and Save in images:
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
#     dicom = dcmread(file)
#     image_array = dicom.pixel_array
#     img = Image.fromarray(image_array)
#     save_path = save_path_images + 'L{0:06}'.format(set_num) + \
#                                    '{0:04}.png'.format(loop)
#     img.save(save_path, 'PNG')
#     loop += 1
#     folder_prev = folder[-2]


# Load labels and save in labels as individual images:
# loop = 0
# set_num = 1
# for file in tqdm(glob.glob(r"D:\BME548Project\Dataset\*\*\*"
#                            r"\Processed\mask.nii.gz",
#                            recursive=True)):
#     img = nib.load(file)
#     array_data = img.get_fdata()
#     for i in range(len(array_data[0, 0, :])-1):
#         image_array = array_data[:, :, i]
#         img = Image.fromarray(image_array)
#         img = img.convert('I')
#         save_path = save_path_labels+'L{0:06}'.format(set_num) + \
#                                      '{0:04}.png'.format(loop)
#         img.save(save_path, 'PNG')
#         sleep(.001)
#         loop += 1
#     loop = 0
#     set_num += 1

print("WOW, you finished!")