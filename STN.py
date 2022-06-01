#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : STN.py
@Date         : 2022/04/29 22:58:12
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : CNN with STN, in order to generate the ground truth for the RPN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

class STN(nn.Module):
    def __init__(self, num_cls) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.localization = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )
        # Regressor for the 2 axis of the box
        self.fc_loc = nn.Sequential(
            nn.Linear(484, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # initialize the weights and biases
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([0.3, 0.7], dtype=torch.float))
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_cls)
        )
            
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x).view(-1, 484)
        bs = x.size(0)
        vertical = x.size(2)
        theta = self.fc_loc(xs)
        theta.clamp_(0, 1)
        # Resize the cropped picture to the same size as the original picture (batch, channel, height, width)
        resize = Resize((vertical, vertical))  # default is bilinear
        for i in range(bs):
            # 只让STN做剪切变换，theta包含y1和y2，对原始图片进行裁剪，y1取小值，y2取大值
            y1 = int(torch.min(theta[i, :], dim=0)[0] * vertical)
            y2 = int(torch.max(theta[i, :], dim=0)[0] * vertical)
            if(y1 == y2):
                y2 = y1 + 10
            x[i, :, :, :] = resize(x[i, :, y1: y2])  # y1是在图片的上方，y2是在图片的下方 
        return x
    
    def forward(self, x):
        stn_out = self.stn(x)
        feature = self.conv_block(stn_out)
        out = self.classifier(feature.view(-1, 64 * 32 * 32))
        return out
    
    def generate_box(self, x):
        '''
        Output: The ground truth box for the RPN, don't resize to the original size
        in the stn block
        '''
        xs = self.localization(x).view(-1, 484)
        vertical = x.size(2)
        theta = self.fc_loc(xs)
        theta.clamp_(0, 1)
        # 只让STN做剪切变换，theta包含y1和y2，对原始图片进行裁剪
        y1 = min(int(theta[0, 0] * vertical), int(theta[0, 1] * vertical))
        y2 = max(int(theta[0, 0] * vertical), int(theta[0, 1] * vertical))
        if y2 == y1:
            y2 = y1 + 10
        # x = x[:, :, y1: y2]  # y1是在图片的上方，y2是在图片的下方, (y1, y2)就确定了box的位置
        return y1, y2  
    
    
    