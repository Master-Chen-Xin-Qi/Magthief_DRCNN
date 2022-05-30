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
app = ["netmusic", "qqmusic", "taobao", "wzry", "wechat", "qq", "weibo", "douyin", "bilibili",
                  "aiqiyi", "subway", "gaode", "baidu", "jingdong", "word"]
app.sort()
CONFIG= {
    "app_name" : app,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "SLIDE_RATE" : 0.15,
    "WINDOW_LEN" : 500,
    "FS" : 100,  
    "BATCH_SIZE" : 1,  # 训练DRCNN的batchsize
    "STN_BATCH_SIZE" : 32,  # 训练STN的batchsize
    "STN_GT_BATCH_SIZE" : 1,  # 产生gt时的batchsize
    "epoch" : 100,
    "STN_epoch" : 500,
    "optimizer" : torch.optim.Adam,
    "STN_optimizer" : torch.optim.Adam,
    "lr" : 1e-4, 
    "STN_lr" : 1e-1,
    "criterion" : torch.nn.CrossEntropyLoss(),
    "STN_criterion" : torch.nn.CrossEntropyLoss(),
    "pics_path" : "./figs",
    "save_model_path" : "./save_checkpoints/DRCNN/best_model.pt",
    "save_STN_path" : "./save_checkpoints/STN/best_stn_model",
    "anchor_scales" : [8, 16, 32],
    "anchor_ratios" : [0.5, 1, 2],
    "feat_stride" : 16,
}

cfg = dict()
cfg= {
    "TRAIN":{
        "RPN_PRE_NMS_TOP_N" : 12000,
        "RPN_POST_NMS_TOP_N" : 2000,
        "RPN_NMS_THRESH" : 0.7,
        "RPN_MIN_SIZE" : 8,
    },
    "TEST":{
        "RPN_PRE_NMS_TOP_N" : 6000,
        "RPN_POST_NMS_TOP_N" : 300,
        "RPN_NMS_THRESH" : 0.7,
        "RPN_MIN_SIZE" : 16,
    }
}