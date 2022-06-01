#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : STN_main.py
@Date         : 2022/06/01 12:52:42
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : train STN to generate the gt for DRCNN
'''

import numpy as np
from config import CONFIG
from train import Trainer, STN_Trainer
from dataset import get_pic_and_labels, get_stn_loader, get_app_loader
from utils import delete_train_his, set_seed
from STN import STN

if __name__ == '__main__':
    set_seed(seed=42)
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    
    # Step 1: train the STN and generate gt for every app
    
    start = 0
    drcnn_labels = []
    for pos_app in CONFIG["app_name"]:
        gt_model = STN(num_cls=2)
        delete_train_his(app=pos_app, STN_flag=True, DRCNN_flag=False)
        train_loader, val_loader, test_loader, app_len = get_stn_loader(pic_names, labels, pos_app)  
        stn_trainer = STN_Trainer(model=gt_model, device=CONFIG["device"], optimizer=CONFIG["STN_optimizer"],
                        lr=CONFIG["STN_lr"], criterion=CONFIG["STN_criterion"], num_epochs=CONFIG["STN_epoch"])

        # train STN to get the gt 
        stn_trainer.run(train_loader, val_loader, pos_app)
        acc = stn_trainer.test(test_loader, pos_app)
        print(f'Test accuracy is: {acc}%')
        
        # generate the gt boxes for RPN
        app_loader = get_app_loader(np.array(pic_names[start:start+app_len]), np.array(labels[start:start+app_len]))
        box_labels = stn_trainer.generate_gt_boxes(app_loader, pos_app)
        # save_gt_pics(gt=train_gt, app_name=pos_app, pic_names=pic_names)
        drcnn_labels.extend(box_labels)
        start = start + app_len
    
    np.save('./drcnn_labels.npy', np.array(drcnn_labels))
    print('Already save all labels!')