import skimage.io
import numpy as np
import SimpleITK as sitk
import cv2
import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from selective_search import selective_search
from addNoise import p_noise

import json


def readImage(filePath, img_number):
    img_path = filePath
    img = skimage.io.imread(img_path)
    example_img = cv2.normalize(img[img_number], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_rgb = cv2.cvtColor(example_img, cv2.COLOR_GRAY2BGR)
    return img_rgb


def getOrSp(filepath):
    itkimage = sitk.ReadImage(filepath)
    OR = itkimage.GetOrigin()
    SP = itkimage.GetSpacing()
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return OR, SP, numpyImage


def getNoduleLoc(uid):
    data = pd.read_csv("D:\machine learning and image processing final\\annotations.csv")
    nodule = []
    for i in range(1186):
        if data.iloc[i, 0] == uid:
            nodule.append([data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]])
    nodule = np.array(nodule)
    return nodule


def findNodule(ct_scan, nodules, Origin, SP):
    nodule_label = []
    img_number = []
    for idx in range(nodules.shape[0]): 
        if abs(nodules[idx, 0]) + abs(nodules[idx, 1]) + abs(nodules[idx, 2]) + abs(nodules[idx, 3]) == 0:
            continue
        x, y, z = int((nodules[idx, 0] - Origin[0]) / SP[0]), int((nodules[idx, 1] - Origin[1]) / SP[1]), int(
            (nodules[idx, 2] - Origin[2]) / SP[2])
        radius = int(nodules[idx, 3] / SP[0] / 2)
        nodule_label.append([x-radius, y-radius, radius * 2, radius * 2])
        img_number.append(z)
    return nodule_label, img_number


def getNodule(filepath):
    OR, SP, image = getOrSp(filepath)
    file_id = re.compile("(1.3.*?).mhd")
    uid = re.findall(file_id, filepath)[0]
    noduleLoc = getNoduleLoc(uid)
    nodule_label, img_number = findNodule(image, noduleLoc, OR, SP)
    return nodule_label, img_number, uid[-4:]


def generateSet(regions):
    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] > 3000:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        if w / h > 1.5 or h / w > 1.5:
            continue
        candidates.add(r['rect'])
    return candidates


def iou(axes, nodule):
    vertice1 = [axes[0], axes[1], axes[0] + axes[2], axes[1] + axes[3]]
    vertice2 = [nodule[0], nodule[1], nodule[0] + nodule[2], nodule[1] + nodule[3]]
    lu = np.maximum(vertice1[0:2], vertice2[0:2])
    rd = np.minimum(vertice1[2:], vertice2[2:])
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]
    square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    return np.clip(inter_square / union_square, 0.0, 1.0)


def store_info(img, axes, nodule):
    info = {
        "array": img.tolist(),
        "axes": axes,
        "label": False}
    if iou(axes, nodule) > 0.3:
        info["label"] = True
    return info


def get_title(image):
    if image['label']:
        return "true"
    else:
        return "false"


def outputFile(images, set_num,cnt):
    string = json.dumps(images)
    with open('D:\machine learning and image processing final\\blure_json_file\\'+set_num+"_"+str(cnt)+".json", 'w')as f:
        f.write(string)


def generatePositiveCase(img_rgb, nodule_loc, images, uid, img_number, cnt):
    x1 = nodule_loc[0]
    y1 = nodule_loc[1]
    a = nodule_loc[2]
    maxb = int(a / math.sqrt(0.3))
    count = 0
    start = cnt
    for b in range(a + 1, maxb + 1):
        newrect = [x1 - (b - a), y1 - (b - a), 2 * b - a, 2 * b - a]
        new_rect_image = img_rgb[newrect[1]:newrect[1] + newrect[3],
                                 newrect[0]:newrect[0] + newrect[2], :]
        for i in range(b - a+1):
            for j in range(b - a+1):
                    img = new_rect_image[i:i+b, j:j+b, :]
                    image_224 = cv2.resize(img, (64, 64))
                    img_loc = [newrect[0]+j, newrect[1]+i, b, b]
                    images[uid + "_" + str(img_number) + "_gp_" + str(start)]=store_info(image_224, img_loc, nodule_loc)
                    count+=1
                    start+=1
                    if count>200:
                        print("generated {} cases by generatePositiveCase".format(count))
                        return count
    print("generated {} cases by generatePositiveCase".format(count))
    return count


def generate_cropped_images(filepath="./1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd", set_num="0", cnt=1):
    images = {}
    nodule_loc, img_number, uid = getNodule(filepath)
    print(img_number)
    count = 1
    gp_num = 0
    for i in range(len(img_number)):
        img_rgb = readImage(filepath, img_number[i])
        img_rgb = p_noise(img_rgb)
        print(nodule_loc[i])
        gp_num = generatePositiveCase(img_rgb, nodule_loc[i], images, uid, img_number[i], gp_num)
        img_lbl, regions = selective_search(
            img_rgb, sigma=0.5, scale=200, min_size=20)
        candidates = generateSet(regions)
        for x, y, w, h in candidates:
            image = img_rgb[y:y + h, x:x + w, :]
            image_224 = cv2.resize(image, (64, 64))
            images[uid + "_" + str(img_number[i]) + "_sr_" + str(count)] = store_info(image_224, [x, y, w, h],
                                                                                   nodule_loc[i])
            count += 1
        print("generated {} cases by selective search".format(count))
    count = 0
    for i in images.keys():
        if images[i]['label']:
            count+=1
    print("positive cases: {}".format(count))
    print(len(images))
    outputFile(images, set_num, cnt)


if __name__ == '__main__':
    generate_cropped_images()
