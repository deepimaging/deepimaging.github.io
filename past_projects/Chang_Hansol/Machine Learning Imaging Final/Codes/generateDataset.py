import json
import numpy as np
from get_example_image import get_file_name


def load_json(filepath):
    images = json.load(open(filepath))
    return images


def generateDataset(jsonfilepath):
    images = load_json(jsonfilepath)
    pre_x_train = []
    pre_y_train = []
    count = 0
    for i in images.keys():
        numpyImage = images[i]["array"]
        pre_x_train.append(numpyImage)
        if images[i]["label"]:
            pre_y_train.append(1)
        else:
            pre_y_train.append(0)
        if images[i]['label']:
            count += 1
    if not len(images):
        return None, None
    if 0.3<count/len(images)<0.7:
        return np.array(pre_x_train), np.array(pre_y_train)
    return None, None


if __name__ == '__main__':
    filedir = "D:\machine learning and image processing final\\blure_json_file"
    name_list = get_file_name(filedir)
    cnt = 1
    x_train = np.array([])
    y_train = np.array([])
    for i in name_list:
        jsonfilepath = filedir + '\\' + i
        print(jsonfilepath)
        new_x_train, new_y_train = generateDataset(jsonfilepath)
        if new_x_train is not None and new_y_train is not None:
            if cnt == 1:
                x_train = new_x_train
                y_train = new_y_train
            else:
                x_train = np.concatenate((x_train, new_x_train), axis=0)
                y_train = np.concatenate((y_train, new_y_train), axis=0)
            cnt+=1
    print(x_train.shape, y_train.shape)
    cnt = 0
    for i in y_train:
        if i:
            cnt+=1
    print("positive case / all case = {}".format(cnt/y_train.size))
    # 3000, 31697 for non noise, 3000, 26309 for noise
    for i in range(1, 10):
        if i < 9:
            train_dict = {"x_train": x_train[3000*(i-1):3000*i,:,:,:].tolist(), "y_train":y_train[3000*(i-1):3000*i].tolist()}
            string = json.dumps(train_dict)
            with open('D:\machine learning and image processing final\\blure_train_json\\' + "train"+str(i)+".json", 'w')as f:
                f.write(string)
        else:
            train_dict = {"x_train": x_train[24000:26309, :, :, :].tolist(),
                          "y_train": y_train[24000:26309].tolist()}
            string = json.dumps(train_dict)
            with open('D:\machine learning and image processing final\\blure_train_json\\' + "train" + str(i) + ".json",
                      'w')as f:
                f.write(string)


