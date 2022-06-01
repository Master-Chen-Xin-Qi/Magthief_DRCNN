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
from config import CONFIG
from train import Trainer, STN_Trainer
from dataset import get_pic_and_labels, get_data_loader
from utils import set_seed, extract_loader
from STN import STN
from DRCNN_modules.DRCNN import DRCNN

if __name__ == '__main__':
    set_seed(seed=42)
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    
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