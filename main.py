#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : main.py
@Date         : 2022/04/15 16:38:43
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Main function for the project
'''

import torch
import numpy as np
from sklearn.metrics import f1_score
# from model import DRCNN
from config import CONFIG
from train import Trainer, STN_Trainer
from dataset import get_pic_and_labels, get_data_loader, get_stn_loader, get_app_loader
from utils import save_gt_pics, delete_train_his, set_seed, extract_loader
from STN import STN
from DRCNN_modules.DRCNN import DRCNN

if __name__ == '__main__':
    set_seed(seed=42)
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    
    # Step1: train the STN and generate gt for every app
    # start = 0
    # drcnn_labels = []
    # for pos_app in CONFIG["app_name"]:
    #     gt_model = STN(num_cls=2)
    #     delete_train_his(app=pos_app, STN_flag=True, DRCNN_flag=False)
    #     train_loader, val_loader, test_loader, app_len = get_stn_loader(pic_names, labels, pos_app)  
    #     stn_trainer = STN_Trainer(model=gt_model, device=CONFIG["device"], optimizer=CONFIG["STN_optimizer"],
    #                     lr=CONFIG["STN_lr"], criterion=CONFIG["STN_criterion"], num_epochs=CONFIG["STN_epoch"])

    #     # train STN to get the gt 
    #     stn_trainer.run(train_loader, val_loader, pos_app)
    #     acc = stn_trainer.test(test_loader, pos_app)
    #     print(f'Test accuracy is: {acc}%')
        
    #     # generate the gt boxes for RPN
    #     app_loader = get_app_loader(np.array(pic_names[start:start+app_len]), np.array(labels[start:start+app_len]))
    #     box_labels = stn_trainer.generate_gt_boxes(app_loader, pos_app)
    #     # save_gt_pics(gt=train_gt, app_name=pos_app, pic_names=pic_names)
    #     drcnn_labels.extend(box_labels)
    #     start = start + app_len
    
    # np.save('./drcnn_labels.npy', np.array(drcnn_labels))
    
    # Step2: train the DRCNN
    
    drcnn_labels = np.load('./drcnn_labels.npy', allow_pickle=True)
    # prepare loader for DRCNN
    train_loader, val_loader, test_loader = get_data_loader(pic_names, drcnn_labels)
    model = DRCNN()
    trainer = Trainer(model=model, device=CONFIG["device"], optimizer=CONFIG["optimizer"], 
                      lr=CONFIG["lr"], criterion=CONFIG["criterion"], num_epochs=CONFIG["epoch"])
    
    trainer.run(train_loader, val_loader, test_loader)
    
    # Step3: evaluate the DRCNN on test set
    model.load_state_dict(torch.load(CONFIG["save_model_path"]))
    test_img, test_labels = extract_loader(test_loader)  # extract the test data and labels
    _bboxes, _labels, _scores = model.predict(test_loader)  # predict result
    f1_score_macro = f1_score(_labels, test_labels, average='macro')
    print(f1_score_macro)