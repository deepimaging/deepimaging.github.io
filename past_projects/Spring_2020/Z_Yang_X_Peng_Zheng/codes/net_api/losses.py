import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F


class DiceCoeff(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        inter = torch.dot(input, target) + 0.0001
        union = torch.sum(input ** 2) + torch.sum(target ** 2) + 0.0001

        t = 2 * inter.float() / union.float()
        return t


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001

        #output = output.exp()
        #print(output)
        _,pred = output.topk(1, dim=1)
        pred = pred.type(torch.LongTensor).cuda()
        # print(pred.shape)
        # print(pred.size()[0])
        encoded_target = output.detach() * 0
        # encoded_output = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target, 1)
            #encoded_output.scatter_(1, pred, 1)

        if weights is None:
            weights = 1
        smooth = 1
        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)+smooth
        #print(numerator)
        denominator = output + encoded_target 

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps + smooth
        #print(denominator)
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

def weighted_log_loss(output,weight,target):
    log_out = -1.0*torch.log(output)
    #print(weight.min())
    encoded_target = output.detach() * 0
    # print(target.unsqueeze(1).shape)
    # print(encoded_target.shape)
    # print(target.shape)
    # print(output.shape)
    encoded_target.scatter_(1, target, 1)

    log_out = log_out * encoded_target
    log_out = torch.sum(log_out,dim=1)
    # print(log_out_1)
    # print(log_out[0,:,1,1])
    loss = torch.mean(log_out*weight)

    return loss

def weighted_log_loss_con(output,weight,target):
    log_out = -1.0*torch.log(output)
    #print(weight.min())
    #encoded_target = output.detach() * 0
    #encoded_target.scatter_(1, target.unsqueeze(1), 1)

    log_out = log_out * target
    log_out = torch.sum(log_out,dim=1)
    # print(log_out_1)
    # print(log_out[0,:,1,1])
    loss = torch.mean(log_out*weight)
    return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target,weight):
        # TODO: why?
        target = target.type(torch.LongTensor).cuda()
        # print(input.shape)
        input_soft = F.softmax(input,dim=1)
        # print(input.shape)
        #connect_out_soft = F.sigmoid(connect_out)
        # print(con_tar.shape)
        # input_class_semi = semi_input_soft[:,0:4,:,:]
        # input_con_semi = semi_input_soft[:,4:12,:,:]
        # print(input_con.shape)
        # print(input_soft.shape)
        # print(target.shape)
        # print(weight.shape)
        loss_out = weighted_log_loss(input_soft,weight,target)
        loss_out_dice = torch.mean(self.dice_loss(input_soft, target))
        #connect_dice = torch.mean(self.dice_loss(connect_out_soft, con_tar.type(torch.LongTensor).cuda()))
        #loss_out_semi = weighted_log_loss(input_class_semi,weight,target)
        # y2 = torch.mean(self.dice_loss(input_soft, target,weights=torch.tensor([1,5,5,5]).cuda()))
        
        #loss_con_semi = weighted_log_loss_con(connect_out_soft,1, con_tar.type(torch.LongTensor).cuda())
        # y1 = self.cross_entropy_loss.forward(input, target)#torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        #y = loss_out_dice  + connect_dice
        #print(loss_out,loss_out_dice,loss_con_semi)

        #print(loss_out.item(),loss_con_semi.item(),loss_out_semi.item(),loss_out_dice.item())
        return loss_out+loss_out_dice, loss_out_dice





def UD_loss(self, class_tp, ground_tp, weight, class_num):
    single_loss = 0
    UD_loss = 0
    count = 0
    for i in range(class_num):
        if(len(ground_tp[i])):
            ground = torch.stack(ground_tp[i], dim=0)
            output = torch.stack(class_tp[i], dim=0)
            soft = torch.nn.functional.softmax(output, dim=1)
            single_loss = float(len(class_tp[i]))*torch.nn.functional.cross_entropy(output, ground)
            single_loss = single_loss*weight[i]
            UD_loss += single_loss

    del ground
    del output
    del single_loss
    #del soft
    for j in range(class_num):
        count += float(len(class_tp[j]))
    #print(count)
    #print(UD_loss)
    UD_loss = UD_loss/count
    #print(UD_loss)
    return UD_loss
