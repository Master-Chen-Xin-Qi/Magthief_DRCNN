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
from train import Trainer
from dataset import get_pic_and_labels, get_data_loader

if __name__ == '__main__':
    model = DRCNN(num_cls=len(CONFIG["app_name"]))
    trainer = Trainer(model=model, device=CONFIG["device"], optimizer=CONFIG["optimizer"], 
                      lr=CONFIG["lr"], criterion=CONFIG["criterion"], num_epochs=CONFIG["epoch"])
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    train_loader, val_loader, test_loader = get_data_loader(pic_names, labels)
    trainer.run(train_loader, val_loader, test_loader)