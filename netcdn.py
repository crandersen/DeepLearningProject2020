# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:53:05 2021

@author: chrisan
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn import Linear, Conv2d, BatchNorm2d, BatchNorm1d, MaxPool2d, Dropout2d, ConvTranspose2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

# hyperameters of the model

channels = 1
height = 512
width = 512

kernel_size = 3 # [height, width]
stride_conv1 = 1 # [stride_height, stride_width]
padding_conv1 = 1

maxP_kernel_size = 2
maxP_stride = 2
maxP_padding = 0

dropoutc = 0.2

def compute_maxP_dim(height, width, kernel_size):
    conv_height = int((height - kernel_size + 2 * padding_conv1) / stride_conv1 + 1)
    conv_width = int((width - kernel_size + 2 * padding_conv1) / stride_conv1 + 1)
    
    maxP_height = int((conv_height - maxP_kernel_size + 2 * maxP_padding) / maxP_stride + 1)
    maxP_width = int((conv_width - maxP_kernel_size + 2 * maxP_padding) / maxP_stride + 1)
    return maxP_height, maxP_width

def conv_layer(in_chan, out_chan):
    layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chan,
                      out_channels=out_chan,
                      kernel_size=kernel_size,
                      stride=stride_conv1, 
                      padding=padding_conv1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxP_kernel_size, stride=maxP_stride, padding=maxP_padding), 
            nn.BatchNorm2d(out_chan)
            #nn.Dropout2d(p=dropoutc)
            )
    return layer

def linear_layer(in_chan, out_chan):
    layer = nn.Sequential(
            nn.Linear(in_features=in_chan, 
                          out_features=out_chan,
                          bias=True), 
            nn.ReLU(),
            nn.BatchNorm1d(out_chan)
            #nn.Dropout()
            )
    return layer


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes

        # Features
        self.maxPool_h, self.maxPool_w = compute_maxP_dim(height, width, kernel_size)
        self.maxPool_h2, self.maxPool_w2 = compute_maxP_dim(self.maxPool_h, self.maxPool_w, kernel_size)
        self.maxPool_h3, self.maxPool_w3 = compute_maxP_dim(self.maxPool_h2, self.maxPool_w2, kernel_size)
        self.maxPool_h4, self.maxPool_w4 = compute_maxP_dim(self.maxPool_h3, self.maxPool_w3, kernel_size)
        

        # Convolutional layer combos
        num_filters_conv = [channels, 32, 64, 64, 64] 

        self.conv_layer1 = conv_layer(num_filters_conv[0], num_filters_conv[1])
        self.conv_layer2 = conv_layer(num_filters_conv[1], num_filters_conv[2])
        self.conv_layer3 = conv_layer(num_filters_conv[2], num_filters_conv[3])
        self.conv_layer4 = conv_layer(num_filters_conv[3], num_filters_conv[4])

        self.l_conv_in_features4 = num_filters_conv[3] * self.maxPool_h4 * self.maxPool_w4


        # Linear layers
        num_l = [self.l_conv_in_features4, 100, 100, 100, 100]

        self.linear_layer1 = linear_layer(num_l[0], num_l[1])
        self.linear_layer2 = linear_layer(num_l[1], num_l[2])
        self.linear_layer3 = linear_layer(num_l[2], num_l[3])
        self.linear_layer4 = linear_layer(num_l[3], num_l[4])

        self.l_out = Linear(in_features=num_l[-1], out_features=num_classes, bias=False)
        
        # add dropout to network
        #self.dropout = Dropout2d(p=0.5)
    
        # Batchnormalization
        self.batchNorm = BatchNorm1d(self.l_conv_in_features4)

    def forward(self, x):
        x = self.conv_layer1(x)
        # print('convlayer1')
        # print(x.shape)
        x = self.conv_layer2(x)
        # print('convlayer2')
        # print(x.shape)
        x = self.conv_layer3(x)
        # print('convlayer3')
        # print(x.shape)
        x = self.conv_layer4(x)
        # print('convlayer4')
        # print(x.shape)
        x = x.view(-1, self.l_conv_in_features4)
        #x = self.dropout(x)                                                     # regulizing 
        # x = self.batchNorm(x)                                                   # normalizing to decrease the runtime of the epochs
        x = self.linear_layer1(x)
        # print('linlayer1')
        # print(x.shape)
        x = self.linear_layer2(x)
        # print('linlayer2')
        # print(x.shape)
        x = self.linear_layer3(x)
        # print('linlayer3')
        # print(x.shape)
        x = self.linear_layer4(x)
        # print('linlayer4')
        # print(x.shape)
        return softmax(self.l_out(x), dim=1)
    

