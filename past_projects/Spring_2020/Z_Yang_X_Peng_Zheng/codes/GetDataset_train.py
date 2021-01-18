# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np

#root = '/home/ziyun/Desktop/Project/Mice_seg/Data_train'
class MyDataset_train(data.Dataset):# 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        img_ls = []
        label_ls = []
        weight_ls = []
        connect_ls = []

        length = len(root)
        for fold_id in range(length):
            # print(file)
            img_list = glob.glob(root[fold_id]+'/original'+'/*.tif')
            for img in img_list:
                index = img.split('original/')[1].split('.tif')[0]
                #print(index)
                label = root[fold_id]+'/label'+'/'+index+'.tif'
                weight = root[fold_id]+'/weight'+'/'+index+'.tif'
                connect = root[fold_id]+'/connectivity'+'/'+index+'.mat'
                img_ls.append(img)
                label_ls.append(label)
                weight_ls.append(weight)
                connect_ls.append(connect)

        self.img_ls = img_ls
        self.label_ls = label_ls
        self.weight_ls = weight_ls
        self.connect_ls = connect_ls
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_root = self.img_ls[index]
        label_root = self.label_ls[index]
        weight_root = self.weight_ls[index]
        connect_root = self.connect_ls[index]
        img = Image.open(img_root)
        mask = imread(label_root)
        weight_map = imread(weight_root)
        connect_mat = scio.loadmat(connect_root)['data']
        #img = np.array(img).astype(np.uint8)
        img = np.array(img)/257.0
    
        # print(img)


        if self.transform is not None:
            img = self.transform(Image.fromarray(img[0:512,:]))  # 是否进行transform

            mask = torch.from_numpy(mask[0:512,:]).unsqueeze(0)
            weight_map = torch.from_numpy(weight_map[0:512,:]).unsqueeze(0)
            connect_root = torch.from_numpy(connect_mat[:,0:512,:])
        return img, mask, weight_map#, connect_root

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.img_ls)


# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
#train_data = MyDataset(root, 'train_label.txt', transform=transforms.ToTensor())

