#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : dataset.py
@Date         : 2022/04/16 16:32:40
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Dataset and dataloader of spectrum figures
'''

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import CONFIG
from utils import label_process

# STN产生gt的dataset
class MyDataset(Dataset):
    def __init__(self, pics_name, labels) -> None:
        super().__init__()
        self.pic_names = pics_name
        self.labels = labels
     
    def __getitem__(self, index):
        pic_file = self.pic_names[index]
        label = self.labels[index]
        pic = Image.open(pic_file).convert('L')  # 转为灰度图
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([60, 60])(pic)  # resize到60*60
        label = torch.tensor(label, dtype=torch.long)
        # unloader = transforms.ToPILImage()
        # image = pic.cpu().clone()
        # image = unloader(image)
        # image.save('example.png')
        return pic, label
    
    def __len__(self):
        return len(self.pic_names)
    
    
class DRCNN_Dataset(Dataset):
    def __init__(self, pics_name, labels) -> None:
        super().__init__() 
        self.pics_name = pics_name
        self.labels = labels
    
    def __getitem__(self, index):
        pic_file = self.pics_name[index]
        box, label = self.labels[index]  # box, label
        pic = Image.open(pic_file).convert('L')  # 转为灰度图
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([60, 60])(pic)  # resize到60*60
        label = torch.tensor(label, dtype=torch.long)
        box = torch.tensor(box, dtype=torch.long)
        return pic, box, label
    
    def __len__(self):
        return len(self.pics_name)
      
# get dataloader for DRCNN
def get_data_loader(pic_names, drcnn_labels):
    train_val_pics, test_pics, train_val_labels, test_labels = \
        train_test_split(pic_names, drcnn_labels, test_size=0.1, shuffle=True)
    train_pics, val_pics, train_labels, val_labels = \
        train_test_split(train_val_pics, train_val_labels, test_size=0.2, shuffle=True)
    train_dataset = DRCNN_Dataset(train_pics, train_labels)
    val_dataset = DRCNN_Dataset(val_pics, val_labels)
    test_dataset = DRCNN_Dataset(test_pics, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    return train_loader, val_loader, test_loader

# get data for baseline model (SVM, RF) training
def get_data(pic_names, labels):
    train_val_pics, test_pics, train_val_labels, test_labels = train_test_split(pic_names, labels, test_size=0.1, shuffle=True)
    train_pics, val_pics, train_labels, val_labels = train_test_split(train_val_pics, train_val_labels, test_size=0.2, shuffle=True)
    train_data, val_data, test_data = np.zeros((1, 3600)), np.zeros((1, 3600)), np.zeros((1, 3600))
    for pic_name in train_pics:
        pic = Image.open(pic_name).convert('L')  # 转为灰度图
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([60, 60])(pic)  # resize到60*60
        pic = pic.numpy().reshape(1, -1)
        train_data = np.vstack((train_data, pic))
    print(f'Train data finish! Len:{len(train_data)}')
    for pic_name in val_pics:
        pic = Image.open(pic_name).convert('L')  # 转为灰度图
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([60, 60])(pic)  # resize到60*60
        pic = pic.numpy().reshape(1, -1)
        val_data = np.vstack((val_data, pic))
    print(f'Val data finish! Len:{len(val_data)}')
    for pic_name in test_pics:
        pic = Image.open(pic_name).convert('L')  # 转为灰度图
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([60, 60])(pic)  # resize到60*60
        pic = pic.numpy().reshape(1, -1)
        test_data = np.vstack((test_data, pic))
    print(f'Train data finish! Len:{len(test_data)}')
    return train_data[1:, :], train_labels, val_data[1:, :], val_labels, test_data[1:, :], test_labels
        
def get_stn_loader(pic_names, labels, pos_app):
    '''
        get dataloader for STN, label is the positive label, and none-app signals are negative labels,
        we train STN as a two-class classifification.
    ''' 
    from copy import deepcopy
    stn_labels = deepcopy(labels)
    pos_label = label_process(CONFIG["app_name"])[pos_app]  # 根据app名称得到正样本的label
    app_len = 0
    # 正样本label为1，没有运行app (或运行其他app)时的label为0
    for i in range(len(labels)):
        if stn_labels[i] == pos_label:
            stn_labels[i] = 1
            app_len += 1
        else:
            stn_labels[i] = 0
    train_val_pics, test_pics, train_val_labels, test_labels = train_test_split(pic_names, stn_labels, test_size=0.1, shuffle=True)
    train_pics, val_pics, train_labels, val_labels = train_test_split(train_val_pics, train_val_labels, test_size=0.2, shuffle=True)
    train_dataset = MyDataset(train_pics, train_labels)
    val_dataset = MyDataset(val_pics, val_labels)
    test_dataset = MyDataset(test_pics, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["STN_BATCH_SIZE"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["STN_BATCH_SIZE"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["STN_BATCH_SIZE"], shuffle=True, drop_last=True)
    return train_loader, val_loader, test_loader, app_len

def get_app_loader(pic_names, labels):
    # 正样本的label为1
    # pos_index = np.where(labels == 1)[0]
    # pos_pics = pic_names[pos_index]
    # pos_labels = labels[pos_index]
    app_dataset = MyDataset(pic_names, labels)
    app_loader = DataLoader(app_dataset, batch_size=CONFIG["STN_GT_BATCH_SIZE"], shuffle=False)
    return app_loader
    
# 每个label一个folder，每个folder下是该类别的所有图片，遍历得到所有文件名和label
def get_pic_and_labels(pics_path):
    pic_names = []
    labels = []
    label_dict = label_process(CONFIG["app_name"])
    for root, dirs, _ in os.walk(pics_path):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                files.sort(key=lambda x: int(x[:-4]))
                for file in files:
                    pic_names.append(os.path.join(root, dir, file))
                labels.extend([label_dict[dir]] * len(files))
    return pic_names, labels
    
def get_pic_and_labels_drcnn(pics_path):
    pic_names = []
    labels = []
    label_dict = label_process(CONFIG["app_name"])
    for root, dirs, _ in os.walk(pics_path):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    pic_names.append(os.path.join(root, dir, file))
                labels.extend([label_dict[dir]] * len(files))
    return pic_names, labels

# if __name__ == '__main__':
#     pics_path = './figs'
#     pic_names, labels = get_pic_and_labels(pics_path)
#     train_loader, val_loader, test_loader = get_data_loader(pic_names, labels)
#     for idx, (data, label) in enumerate(train_loader):
#         print(data.shape)
#     print(f"train_loader: {len(train_loader)} val_loader: {len(val_loader)} \
#           test_loader: {len(test_loader)}")