from random import shuffle
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from net_api.losses import CombinedLoss, DiceLoss
from torch.optim import lr_scheduler
import os
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from skimage.io import imread, imsave
if not os.path.exists('visualization'):
    os.makedirs('visualization')

save = 'save'
if not os.path.exists(save):
    os.makedirs(save)
noisy = 'noisy'
if not os.path.exists(noisy):
    os.makedirs(noisy)
class AddGaussianNoise(object):
    """"
    Add Gaussian noise to a batch
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        x = x.cuda()
        noise = torch.randn(x.shape).mul_(self.sigma/255.0).cuda()
        # print('noise shape', noise.shape)
        #noise = noise/torch.max(torch.abs(noise))
        y = x + noise
        #y = (y-torch.min(y))/(torch.max(torch.abs(y))-torch.min(y))
        return y, noise
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    dice_per_class = []
    # print(y_true.shape)
    # print(y_pred.shape)
    for i in range(num_class):
        Pred = y_pred[:,i,:,:]
        GT = y_true[:,i,:,:]
        inter = np.sum(GT* Pred) 
        #print(inter)

        union = np.sum(GT) + np.sum(Pred) + 0.0001
        #print(union)
        t = 2 * inter / union
        #print(t)
        avg_dice = avg_dice + (t / num_class)
        dice_per_class.append(t)
    return avg_dice, dice_per_class


def create_exp_directory(exp_id):
    if not os.path.exists('models/' + str(exp_id)):
        os.makedirs('models/' + str(exp_id))


class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 5e-4,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.0001}

    # default_optim_args = {"lr": 0.05,
    #                       "momentum":0.9,
    #                       "weight_decay": 0.0001}

    gamma = 0.2
    step_size = 20

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=CombinedLoss()):
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.NumClass = 4
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader,exp_id, num_epochs=10, log_nth=5, exp_dir_name='exp_default'):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,
                                        gamma=self.gamma)  # decay LR by a factor of 0.5 every 5 epochs

        
        iter_per_epoch = 1
        # iter_per_epoch = len(train_loader)

        model.cuda()

        print('START TRAIN.')
        curr_iter = 0
        tensor_bd = 'visualization/'+str(exp_id)
        if not os.path.exists(tensor_bd):
            os.makedirs(tensor_bd)
        writer = SummaryWriter(tensor_bd)
        create_exp_directory(exp_id)
        self.noise_gen = AddGaussianNoise(sigma = 40)
        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, test_dice \n')

        # exp_save ='models/' + exp_dir_name
        # if not os.path.exists(exp_save):
        #     os.makedirs(exp_save)

        for epoch in range(num_epochs):
            self._reset_histories()
            model.train()
            scheduler.step()

            for i_batch, sample_batched in enumerate(train_loader):
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])
                w = Variable(sample_batched[2])
                #print(y.max(),w.max())
                #connect = Variable(sample_batched[3])

                # print(X.shape)
                # print(y.shape)
                # print(w.shape)
                # print(connect.shape)
                #print(connect.shape)
                #print(y.shape)
                curr_iter = curr_iter + 1
                    #X, y, w = X.cuda(), y.cuda(), w.cuda()
                w = w.cuda()
                X= X.cuda()
                X, noise = self.noise_gen(X)
                # for i in range(X.shape[0]):
                #     img_add = 'noisy/'+str(i_batch)+'_'+str(i)+'.png'
                #     print(img_add,X[i].shape)
                #     imsave(img_add,X[i].squeeze().cpu().numpy())

                #connect = connect.cuda()
                #print(X)
                y = y.long().cuda()
                #y = y.squeeze()
                # print(y.max())
                # print(w.max())
                optim.zero_grad()
                output,_ = model(X)
                
                loss, dice_loss = self.loss_func(output, y, w)
                # loss = Variable(loss, requires_grad = True)
                self.train_loss_history.append(loss.item())
                loss.backward()
                optim.step()
                
                #writer.flush()
                
                #imsave('/home/ziyun/Desktop/111.png',np.array(X.squeeze()[0].cpu().detach()))
                #entropy =1
                #dice =1
                
                print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f dice:%.3f' %(
                    loss.item(),dice_loss))



                # if i_batch == 30:
                #     pred = get_mask(output)

                #     pred_one_hot = one_hot(pred,[4,self.NumClass,1024,1024])
                #     target_one_hot = one_hot(y.unsqueeze(1),[4,self.NumClass,1024,1024])

                #     dice = per_class_dice(pred_one_hot,target_one_hot,self.NumClass)

                    # writer.add_scalar('Loss',loss.item(), epoch)
                    # writer.flush()

                    # writer.add_scalar('dice',dice, epoch)
                    # writer.flush()
                
                    # grid1 = utils.make_grid(pred)
                    # writer.add_image('Prediction', grid1, global_step=0)
                    # grid2 = utils.make_grid(y.unsqueeze(1))
                    # gt = y[0].unsqueeze(0)
                    # writer.add_image('Ground Turth', grid2, global_step=0)
                    # grid3 = utils.make_grid(X)

                    # writer.add_image('Input', grid3, global_step=0)
            writer.add_scalar('Loss',np.mean(self.train_loss_history), global_step=epoch)
            # writer2.add_scalar('Loss',np.mean(self.train_loss_history)+1, global_step=epoch)
            self.test_epoch(model,val_loader,epoch,exp_id,writer)
            
            print('[Epoch :%d] total loss:%.3f ' %(epoch,loss.item()))
            torch.save(model.state_dict(), 'models/' + str(exp_id) + '/relaynet_epoch' + str(epoch + 1)+'.pth')

        writer.close()
        print('FINISH.')

    def test_epoch(self,model,loader,epoch,exp_id,writer):
        dice_history = []
        NFL = []
        OPL = []
        RPE = []
        BG = []
        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                #w = Variable(test_data[2])
                #w = w.cuda()
                X_test= X_test.cuda()
                #print(X)
                y_test = y_test.long().cuda()
                # print(y.max())
                
                # print(w.max())
                X_test, noise = self.noise_gen(X_test)
                # for i in range(X_test.shape[0]):
                #     img_add = 'noisy/'+str(j_batch)+'_'+str(i)+'.png'
                #     # print(img_add,X[i].shape)
                #     imsave(img_add,X_test[i].squeeze().cpu().numpy())
                output_test, denoised = model(X_test)
                # for i_2 in range(X_test.shape[0]):
                #     img_add_denoise = 'noisy/'+str(j_batch)+'_'+str(i_2)+'_denoised.png'
                #     # print(img_add,X[i].shape)
                #     imsave(img_add_denoise,denoised[i].squeeze().cpu().numpy())
                #print(len(loader))
                #output_test_semi = output_test_semi[:,0:4,:,:]
                pred_test = get_mask(output_test)
                #print(pred_test.shape)
                pred_one_hot_test = one_hot(pred_test,[pred_test.size()[0],9,496,768])
                target_one_hot_test = one_hot(y_test,[y_test.size()[0],9,496,768])

                dice_test,dice_per_class = per_class_dice(pred_one_hot_test,target_one_hot_test,9)
                dice_history.append(dice_test)
                # BG.append(dice_per_class[0])
                # NFL.append(dice_per_class[1])
                # OPL.append(dice_per_class[2])
                # RPE.append(dice_per_class[3])
                grid_denoise = utils.make_grid(denoised)
                grid1_test = utils.make_grid(pred_test)
                writer.add_image('Denoised', grid_denoise, global_step=epoch)
                writer.add_image('Prediction', grid1_test, global_step=epoch)
                grid2_test = utils.make_grid(y_test)
                writer.add_image('Test Ground Turth', grid2_test, global_step=epoch)
                grid3_test = utils.make_grid(X_test)
                writer.add_image('Test Input', grid3_test, global_step=epoch)
                #writer.flush()


                # pred_test_semi = get_mask(output_test_semi)

                # grid1_semi_test = utils.make_grid(pred_test_semi)
                # writer.add_image('Semi Prediction', grid1_semi_test, global_step=epoch)
                #writer.flush()
            

                print('test [Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] dice:%.3f' %(
                    dice_test))
            writer.add_scalar('test dice',np.mean(dice_history), global_step = epoch)
            # writer.add_scalar('NFL',np.mean(NFL), global_step = epoch)
            # writer.add_scalar('OPL',np.mean(OPL), global_step = epoch)
            # writer.add_scalar('RPE',np.mean(RPE), global_step = epoch)
            # writer.add_scalar('BG',np.mean(BG), global_step = epoch)
            # writer.add_scalar('test dice',np.mean(dice_history), global_step = epoch)
            # writer.add_scalar('test dice',np.mean(dice_history), global_step = epoch)

            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(save, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    (epoch + 1),
                    np.mean(dice_history)
                    
                ))


def get_mask(output):
    output = F.softmax(output,dim=1)
    _,pred = output.topk(1, dim=1)
    #pred = pred.squeeze()
    
    return pred

def one_hot(target,shape):

    one_hot_mat = torch.zeros(shape).cuda()

    one_hot_mat.scatter_(1, target, 1)
    return one_hot_mat
