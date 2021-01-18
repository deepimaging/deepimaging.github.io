import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from GetDataset import MyDataset
import relay_net
import glob
from torchvision import datasets, transforms
from relay_net import ReLayNet
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn
from GetDataset_train import MyDataset_train
#torch.set_default_tensor_type('torch.FloatTensor')

torch.cuda.set_device(1)
root = '/home/ziyun/Desktop/Project/Roarke/Chiu/'

file_list = glob.glob(root+'/*')
# file_list_test = glob.glob(root_test+'/*')
file = file_list
# print(len(file))

root = '/home/ziyun/Desktop/Project/Roarke/Chiu/'

file_list = glob.glob(root+'/*')
# file_list_test = glob.glob(root_test+'/*')
file = file_list
# print(len(file))

test_index = np.array([1,5,8,9])
for exp_id in range(10):
    train_index = [i for i in range(10)]
    test_add = []
    train_add = []
    test_add.append(file[test_index[exp_id]])

    train_index = list(set(train_index)-set([test_index[exp_id]]))

    for j in range(len(train_index)):
        train_add.append(file[train_index[j]])


    transform = test_transforms = transforms.Compose([
    	transforms.Grayscale(num_output_channels=1),
    	#transforms.Resize((512,740)),
            transforms.ToTensor()
        ])
    train_data = MyDataset(root = train_add,transform=transform)
    test_data = MyDataset(root = test_add,transform=transform)
    print("Train size: %i" % len(train_data))


    train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=2, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=2, shuffle=False, num_workers=4)

    param ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':3,
            'kernel_w':7,
            'kernel_c': 1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':9
        }

    exp_dir_name = 'Exp01'

    relaynet_model = ReLayNet(param).cuda()
    # if exp_id ==0:
    #     relaynet_model.load_state_dict(torch.load('/home/ziyun/Desktop/Project/Roarke/daily/4.21/11/turn1_fix_net/best_model_output/0/relaynet_epoch9.pth'))
    
    # if exp_id ==1:
    #     relaynet_model.load_state_dict(torch.load('/home/ziyun/Desktop/Project/Roarke/daily/4.21/11/turn1_fix_net/best_model_output/1/relaynet_epoch2.pth'))

    # if exp_id ==2:
    #     relaynet_model.load_state_dict(torch.load('/home/ziyun/Desktop/Project/Roarke/daily/4.21/11/turn1_fix_net/best_model_output/2/relaynet_epoch28.pth'))

    # if exp_id ==3:
    #     relaynet_model.load_state_dict(torch.load('/home/ziyun/Desktop/Project/Roarke/daily/4.21/11/turn1_fix_net/best_model_output/3/relaynet_epoch45.pth'))
    # if torch.cuda.device_count()>1:
    #         relaynet_model=nn.DataParallel(relaynet_model).cuda()

    solver = Solver(optim_args={"lr": 5e-3})

    solver.train(relaynet_model, train_loader, val_loader,exp_id, log_nth=1, num_epochs=80, exp_dir_name=exp_dir_name)


    # SEG_LABELS_LIST = [
    #     {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    #     {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    #     {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    #     {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    #     {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    #     {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    #     {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    #     {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    #     {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    #     {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];
    #     #{"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];
        
    # def label_img_to_rgb(label_img):
    #     label_img = np.squeeze(label_img)
    #     labels = np.unique(label_img)
    #     label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    #     label_img_rgb = np.array([label_img,
    #                               label_img,
    #                               label_img]).transpose(1,2,0)
    #     for l in label_infos:
    #         mask = label_img == l['id']
    #         label_img_rgb[mask] = l['rgb_values']

    #     return label_img_rgb.astype(np.uint8)

    #     import matplotlib.pyplot as plt


    # relaynet_model =  torch.load('models/Exp01/relaynet_epoch20.model')
    # out = relaynet_model(Variable(torch.Tensor(test_data.X[0:1]).cuda(),volatile=True))
    # out = F.softmax(out,dim=1)
    # max_val, idx = torch.max(out,1)
    # idx = idx.data.cpu().numpy()
    # idx = label_img_to_rgb(idx)
    # plt.imshow(idx)
    # plt.show()

    # img_test = test_data.X[0:1]
    # img_test = np.squeeze(img_test)
    # plt.imshow(img_test)
    # plt.show()
