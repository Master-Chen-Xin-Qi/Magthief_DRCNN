#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : main.py
@Date         : 2022/04/15 16:38:43
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Main function for the project
'''

from model import DRCNN
from config import CONFIG
from train import Trainer, STN_Trainer
from dataset import get_pic_and_labels, get_data_loader
from STN import STN

if __name__ == '__main__':
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    train_loader, val_loader, test_loader = get_data_loader(pic_names, labels)
    
    model = DRCNN(num_cls=len(CONFIG["app_name"]))
    gt_model = STN(num_cls=len(CONFIG["app_name"]))
    
    stn_trainer = STN_Trainer(model=gt_model, device=CONFIG["device"], optimizer=CONFIG["STN_optimizer"],
                    lr=CONFIG["STN_lr"], criterion=CONFIG["STN_criterion"], num_epochs=CONFIG["STN_epoch"])

    # get the ground truth for all data
    train_gt, val_gt, test_gt = stn_trainer.run(train_loader, val_loader, test_loader)
    trainer = Trainer(model=model, device=CONFIG["device"], optimizer=CONFIG["optimizer"], 
                      lr=CONFIG["lr"], criterion=CONFIG["criterion"], num_epochs=CONFIG["epoch"])
    
    trainer.run(train_loader, val_loader, test_loader)