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
            nn.Linear(32, 3 * 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_cls)
        )
            
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 11 * 11)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # 只让STN做剪切变换，其变换矩阵是[[1, 0, m], [0, 1, n]]
        theta[:, 0, 0] = 1
        theta[:, 0, 1] = 0
        theta[:, 1, 1] = 1
        theta[:, 1, 0] = 0
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        from PIL import Image
        import torchvision.transforms as transforms
        pics = transforms.ToPILImage()(x.squeeze(0)[0, :, :])
        pics.save('stn_out.png')
        img = Image.open('stn_out.png')
        img.show()
        return x
    
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
    
    
    