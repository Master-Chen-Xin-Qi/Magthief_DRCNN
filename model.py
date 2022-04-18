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
    def __init__(self, input_size, num_cls) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_cls = num_cls
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        
    def forward(self, x):
        
