#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : main.py
@Date         : 2022/04/15 16:38:43
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Main function for the project
'''

import numpy as np
from model import DRCNN
from config import CONFIG
from train import Trainer, STN_Trainer
from dataset import get_pic_and_labels, get_data_loader, get_stn_loader, get_app_loader
from utils import save_gt_pics
from STN import STN

if __name__ == '__main__':
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
       
    # model = DRCNN(num_cls=len(CONFIG["app_name"]))
    gt_model = STN(num_cls=len(CONFIG["app_name"]))
    
    # Step1: train the STN and generate gt for every app
    for pos_app in CONFIG["app_name"]:
        train_loader, val_loader, test_loader = get_stn_loader(pic_names, labels, pos_app)  
        stn_trainer = STN_Trainer(model=gt_model, device=CONFIG["device"], optimizer=CONFIG["STN_optimizer"],
                        lr=CONFIG["STN_lr"], criterion=CONFIG["STN_criterion"], num_epochs=CONFIG["STN_epoch"])

        # train STN to get the gt 
        # stn_trainer.run(train_loader, val_loader, test_loader)
        
        # generate the gt boxes for RPN
        app_loader = get_app_loader(np.array(pic_names), np.array(labels))
        train_gt = stn_trainer.generate_gt_boxes(app_loader)
        save_gt_pics(gt=train_gt, app_name=pos_app, pic_names=pic_names)
        
    # Step2: train the DRCNN
    
    # prepare loader for DRCNN
    # train_loader, val_loader, test_loader = get_data_loader(pic_names, labels)
    
    # trainer = Trainer(model=model, device=CONFIG["device"], optimizer=CONFIG["optimizer"], 
    #                   lr=CONFIG["lr"], criterion=CONFIG["criterion"], num_epochs=CONFIG["epoch"])
    
    # trainer.run(train_loader, val_loader, test_loader)