# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:10:20 2020

@author: cwriz
"""
import glob
from skimage import io
import numpy as np
import cv2
import csv
from tifffile import imread

#get folders (subdirectories) in file below (top level directory)
file_list = glob.glob('C:/Users/cwriz/Documents/duke/Di_Talia/003_minimum_spanning_tree_cluster/003_minimum_spanning_tree_cluster/ROIs/forChristian/*/')
big_list = []
for x in file_list:    #loop thru folders

    if('movie' in x):    #if movie in folder name
        y = glob.glob(x+'cleaned/*')    #get all files in subdirectory cleaned from folder
        for j in y:    #loop thru each file in cleaned folder
            if('ROI3' in j):    #if file is a ROI file, open it
                temp_img = io.imread(j)    #read file
                #temp_img = cv2.imread(j)
                for i in np.arange(temp_img.shape[0]):    #loop thru each slice in stack
                    t_img = temp_img[i,:,:]    #get slice
                    #t_img = t_img.reshape(t_img.shape[1],t_img.shape[0])
                    #print(t_img.shape)    #for debugging
                    #temp_img = np.reshape(temp_img, [temp_img.shape[]])
                    #print(temp_img.shape)
                    t_img = np.array(t_img)
                    #t_img = t_img > 0
                    t_img = t_img.astype(np.uint8)
                    
                    print(type(t_img))
                    contours, hierarchy = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]    #get contours
                    for temp in contours:    #loop thru contours in image
                        ecks, why, width, height = cv2.boundingRect(temp)    #get bounding box for object
                        cv2.rectangle(t_img, (ecks,why),(ecks+width, why+height),(255,0,0),2)    #for drawing bounding box on image
                        #path/to/image.jpg,x1,y1,x2,y2,class_name
                        path_name = j.replace('ROI3','ch1')    #do this because the actual image used for training is a different image
                        path_name = path_name.replace('.tif','')
                        path_name = path_name.replace('\\','/')
                        if(i+1<10):    #do this because im going to split stacks in imagej
                            #temp_list = [path_name+'-000'+str(i+1)+'.tif',str(ecks), str(why), str(ecks+width), str(why),str(ecks+width),str(why+height),str(ecks),str(why+height), 'scale']
                            temp_list = [path_name+'-000'+str(i+1)+'.tif',str(ecks), str(why), str(ecks+width),str(why+height), 'scale']
                        elif(i+1< 100):
                            #temp_list = [path_name+'-00'+str(i+1)+'.tif',str(ecks), str(why), str(ecks+width), str(why),str(ecks+width),str(why+height),str(ecks),str(why+height), 'scale']
                            temp_list = [path_name+'-00'+str(i+1)+'.tif',str(ecks), str(why),str(ecks+width),str(why+height), 'scale']
                        else:
                            #temp_list = [path_name+'-0'+str(i+1)+'.tif',str(ecks), str(why), str(ecks+width), str(why),str(ecks+width),str(why+height),str(ecks),str(why+height), 'scale']
                            temp_list = [path_name+'-0'+str(i+1)+'.tif',str(ecks), str(why), str(ecks+width), str(why+height), 'scale']
                        big_list.append(temp_list)
                    cv2.imshow(j+str(i), t_img)
                    cv2.waitKey(0)
                        

    
with open('C:/Users/cwriz/Documents/duke/Di_Talia/003_minimum_spanning_tree_cluster/003_minimum_spanning_tree_cluster/ROIs/forChristian/'+'big_listx4.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(big_list)