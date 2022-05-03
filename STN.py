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
            nn.Conv2d(1, 8, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 11 * 11, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.LayerNorm(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_cls)
        )
            
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x).view(-1, 10 * 11 * 11)
        vertical = x.size(2)
        theta = self.fc_loc(xs)
        theta.clamp_(0, 1)
        # 只让STN做剪切变换，theta包含y1和y2，对原始图片进行裁剪
        y1 = min(int(theta[0, 0] * vertical), int(theta[0, 1] * vertical))
        y2 = max(int(theta[0, 0] * vertical), int(theta[0, 1] * vertical))
        x = x[:, :, y1: y2]  # y1是在图片的上方，y2是在图片的下方
        
        # for show
        # from PIL import Image
        # import torchvision.transforms as transforms
        # pics = transforms.ToPILImage()(x.squeeze(0)[0, :, :])
        # pics.save('stn_out.png')
        # img = Image.open('stn_out.png')
        # img.show()
        
        # Resize the cropped picture to the same size as the original picture (batch, channel, height, width)
        resize = Resize((vertical, vertical))  # default is bilinear
        out = resize(x)
        return out
    
    def forward(self, x):
        stn_out = self.stn(x)
        feature = self.conv_block(stn_out)
        out = self.classifier(feature.view(-1, 64 * 32 * 32))
        return F.log_softmax(out, dim=1)
    
    def generate_box(self, x):
        '''
        Output: The ground truth box for the RPN
        '''
        feature = self.conv_block(x)
        stn_out = self.stn(feature)
        return stn_out
    
    
    