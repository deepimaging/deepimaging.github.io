# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:49:48 2020

@author: cwriz
"""

import glob
from skimage import io, transform
import numpy as np
import sys
import cv2
import csv


#get folders (subdirectories) in file below (top level directory)
file_list = glob.glob(r'C:\Users\cwriz\Documents\duke\Di_Talia\003_minimum_spanning_tree_cluster\003_minimum_spanning_tree_cluster\ROIs\forChristian\H2A_ERKKTR_01feb19_movie\cleaned\*')
#print(file_list)
for x in file_list:    #loop thru folders

    if('_ch1' in x):    #if movie in folder name
        temp_img = io.imread(x)    #read filetemp_img = io.imread(j)    #read file
        io.imshow(temp_img)
        #temp_img = transform.resize(temp_img,(1000,1000))
        #io.imsave(x, temp_img)
                        
                        

    