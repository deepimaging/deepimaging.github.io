"""ClassificationCNN"""
import torch
import torch.nn as nn
import math
from net_api import sub_module as sm
from torch.autograd import Variable
import torch.nn.functional as F




class gaussian_kernel(nn.Module):
    def __init__(self, kernel_size=3, sigma=2, channels=1):
        super(gaussian_kernel, self).__init__()
        # self.sigma = sigma
        self.kernel_size = kernel_size
        self.channels = channels
        # self.sigma = torch.rand(1)
        self.sigma = torch.FloatTensor([1])
        # print(sigma)
        self.sigma = nn.Parameter(self.sigma,requires_grad =False)
        #self.sigma = Variable(self.sigma,requires_grad =True)
        x_coord = torch.arange(self.kernel_size)
        x_grid = x_coord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (self.kernel_size - 1)/2.
        variance = self.sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)

        with torch.no_grad():
            self.gaussian_kernel = (1./(2.*math.pi*variance)) *\
                              torch.exp(
                                  -torch.sum((xy_grid - mean)**2., dim=-1) /\
                                  (2*variance)
                              )

                # Make sure sum of values in gaussian kernel equals 1.
            self.gaussian_kernel = self.gaussian_kernel / torch.sum(self.gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            self.gaussian_kernel = self.gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
            self.gaussian_kernel = self.gaussian_kernel.repeat(self.channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False,padding =1)

        self.gaussian_filter.weight.data = self.gaussian_kernel
        # self.gaussian_filter.weight.requires_grad = False

    def get_gaussian_k(self,kernel_size=3, sigma=2, channels=1):
        
        x_coord = torch.arange(kernel_size).cuda()
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).cuda()
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().cuda()

        mean = (kernel_size - 1)/2.
        variance = self.sigma**2.
        #variance = nn.Parameter(variance,requires_grad =True)
        #variance = Variable(variance,requires_grad =True)
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # with torch.no_grad():
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance)
                          )
        print(self.sigma)

            # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel =gaussian_kernel.view(1, 1, kernel_size,kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        # gaussian_kernel = Variable(gaussian_kernel,requires_grad = True)
        return gaussian_kernel.cuda()

    def forward(self, x):
        # sigma = torch.FloatTensor(2)
        # sigma = Variable
        return F.conv2d(input = x,weight = self.get_gaussian_k(),padding = 1)
        # return self.gaussian_filter(x)


# def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
#     # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#     x_coord = torch.arange(kernel_size)
#     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#     y_grid = x_grid.t()
#     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

#     mean = (kernel_size - 1)/2.
#     variance = sigma**2.

#     # Calculate the 2-dimensional gaussian kernel which is
#     # the product of two gaussian distributions for two different
#     # variables (in this case called x and y)
#     gaussian_kernel = (1./(2.*math.pi*variance)) *\
#                       torch.exp(
#                           -torch.sum((xy_grid - mean)**2., dim=-1) /\
#                           (2*variance)
#                       )

#     # Make sure sum of values in gaussian kernel equals 1.
#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

#     # Reshape to 2d depthwise convolutional weight
#     gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
#     gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

#     gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
#                                 kernel_size=kernel_size, groups=channels, bias=False)

#     gaussian_filter.weight.data = gaussian_kernel
#     gaussian_filter.weight.requires_grad = False
    
#     return gaussian_filter


class ReLayNet(nn.Module):
    """
    A PyTorch implementation of ReLayNet
    Coded by Shayan and Abhijit

    param ={
        'num_channels':1,
        'num_filters':64,
        'num_channels':64,
        'kernel_h':7,
        'kernel_w':3,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':10
    }

    """

    def __init__(self, params):
        super(ReLayNet, self).__init__()

        self.gaussian_layer = gaussian_kernel()

        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        # params['num_channels'] = 64  # This can be used to change the numchannels for each block
        self.encode3 = sm.EncoderBlock(params)
        self.encode4 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        self.decode4 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        #print(input.shape)
        #print(self.gaussian_layer.gaussian_filter.weight)
        x = self.gaussian_layer(input)
        #print(x.shape)
        # with torch.no_grad():
        e1, out1, ind1 = self.encode1.forward(x)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)
        bn = self.bottleneck.forward(e4)

        d4 = self.decode1.forward(bn, out4, ind4)
        d3 = self.decode2.forward(d4, out3, ind3)
        d2 = self.decode3.forward(d3, out2, ind2)
        d1 = self.decode4.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)
        # for name,param in self.classifier.named_parameters():
        #     print(param)
        # print(prob.shape)
        return prob, x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
