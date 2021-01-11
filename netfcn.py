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
height = 256
width = 256

kernel_size = 3 # [height, width]
stride_conv1 = 1 # [stride_height, stride_width]
padding_conv1 = 1

maxP_kernel_size = 2
maxP_stride = 2
maxP_padding = 0

up_kernel_size = 2
up_stride = 2
up_padding = 0

def compute_conv_dim(height, width, kernel_size):
    conv_height = int((height - kernel_size + 2 * padding_conv1) / stride_conv1 + 1)
    conv_width = int((width - kernel_size + 2 * padding_conv1) / stride_conv1 + 1)
    
    return conv_height, conv_width 

def compute_maxP_dim(conv_height, conv_width, kernel_size):
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
            nn.BatchNorm2d(out_chan)
            )
    return layer

def pool_layer():
    layer = nn.Sequential(
        nn.MaxPool2d(kernel_size=maxP_kernel_size, stride=maxP_stride, padding=maxP_padding)
        )
    return layer

def upsample_layer(in_chan,out_chan):
    layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan,
                                out_channels=out_chan,
                                kernel_size=up_kernel_size,
                                stride=up_stride,
                                padding=up_padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_chan)
        )
    return layer


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes

        # Features
        self.conv_h, self.conv_w = compute_conv_dim(height,width,kernel_size)
        self.maxPool_h, self.maxPool_w = compute_maxP_dim(self.conv_h, self.conv_w, kernel_size)
        self.conv_h2, self.conv_w2 = compute_conv_dim(self.maxPool_h,self.maxPool_w,kernel_size)
        self.maxPool_h2, self.maxPool_w2 = compute_maxP_dim(self.conv_h2, self.conv_w2, kernel_size)
        self.conv_h3, self.conv_w3 = compute_conv_dim(self.maxPool_h2,self.maxPool_w2,kernel_size)
        self.maxPool_h3, self.maxPool_w3 = compute_maxP_dim(self.conv_h3, self.conv_w3, kernel_size)
        self.conv_h3, self.conv_w3 = compute_conv_dim(self.maxPool_h3,self.maxPool_w3,kernel_size)
        

        # Convolutional layer combos
        num_filters_conv = [channels, 32, 64, 128, 256,self.num_classes] 

        self.conv_layer01 = conv_layer(num_filters_conv[0], num_filters_conv[1])
        self.conv_layer1 = conv_layer(num_filters_conv[1], num_filters_conv[1])
        self.conv_layer12 = conv_layer(num_filters_conv[1], num_filters_conv[2])
        self.conv_layer2 = conv_layer(num_filters_conv[2], num_filters_conv[2])
        self.conv_layer23 = conv_layer(num_filters_conv[2], num_filters_conv[3])
        self.conv_layer3 = conv_layer(num_filters_conv[3], num_filters_conv[3])
        self.conv_layer34 = conv_layer(num_filters_conv[3], num_filters_conv[4])
        self.conv_layer4 = conv_layer(num_filters_conv[4], num_filters_conv[4])
        self.conv_layer43 = upsample_layer(num_filters_conv[4], num_filters_conv[3])
        self.conv_layer32 = upsample_layer(num_filters_conv[3], num_filters_conv[2])
        self.conv_layer21 = upsample_layer(num_filters_conv[2], num_filters_conv[1])
        self.conv_layer10 = conv_layer(num_filters_conv[1], num_filters_conv[5])
        
        self.pool_layer12 = pool_layer()
        self.pool_layer23 = pool_layer()
        self.pool_layer34 = pool_layer()

        self.l_conv_in_features = num_filters_conv[5] * self.conv_h * self.conv_w

        
        # add dropout to network
        #self.dropout = Dropout2d(p=0.5)
    
        # Batchnormalization
        self.batchNorm = BatchNorm1d(self.l_conv_in_features)

    def forward(self, x):
        x = self.conv_layer01(x)
        y = x
        for i in range(3):
            x = self.conv_layer1(x)
        x = x.add(y)
        x = self.conv_layer1(x)
        x_down1 = x
        
        x = self.pool_layer12(x)
        x = self.conv_layer12(x)
        y = x
        for i in range(3):
            x = self.conv_layer2(x)
        x = x.add(y)
        x = self.conv_layer2(x)
        x_down2 = x
        
        x = self.pool_layer23(x)
        x = self.conv_layer23(x)
        y = x
        for i in range(3):
            x = self.conv_layer3(x)
        x = x.add(y)
        x = self.conv_layer3(x)
        x_down3 = x
        
        x = self.pool_layer34(x)
        x = self.conv_layer34(x)
        y = x
        for i in range(3):
            x = self.conv_layer4(x)
        x = x.add(y)
        x = self.conv_layer4(x)
        
        x = self.conv_layer43(x)
        x = x.add(x_down3)
        x = self.conv_layer3(x)
        y = x
        for i in range(3):
            x = self.conv_layer3(x)
        x = x.add(y)
        x = self.conv_layer3(x)
        
        x = self.conv_layer32(x)
        x = x.add(x_down2)
        x = self.conv_layer2(x)
        y = x
        for i in range(3):
            x = self.conv_layer2(x)
        x = x.add(y)
        x = self.conv_layer2(x)
        
        x = self.conv_layer21(x)
        x = x.add(x_down1)
        x = self.conv_layer1(x)
        y = x
        for i in range(3):
            x = self.conv_layer1(x)
        x = x.add(y)
        x = self.conv_layer10(x)
        
        return softmax(x, dim=1)
    

