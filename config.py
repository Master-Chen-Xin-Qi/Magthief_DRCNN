#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : config.py
@Date         : 2022/04/15 16:38:15
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Basic parameters for the project
'''
import torch

CONFIG= {
    "app_name" : ["netmusic", "taobao", "wzry"],
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "SLIDE_RATE" : 0.15,
    "WINDOW_LEN" : 300,
    "FS" : 50,
    "BATCH_SIZE" : 1,
    "epoch" : 100,
    "optimizer" : torch.optim.Adam,
    "lr" : 1e-4, 
    "criterion" : torch.nn.CrossEntropyLoss,
    "pics_path" : "./figs",
    "save_model_path" : "./save_checkpoints/best_model.pth",
}