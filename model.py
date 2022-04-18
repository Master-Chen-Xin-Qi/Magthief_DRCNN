#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : model.py
@Date         : 2022/04/16 16:17:51
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : The DRCNN model of this project 
'''

import torch
import torch.nn as nn



class DRCNN(nn.Module):
    def __init__(self, num_cls) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        
    def forward(self, x):
        '''
            x: (batch_size, 1, 60, 60)
        '''
        feature = self.conv_layer(x)  # (batch_size, 64, 32, 32)
        return feature
        
