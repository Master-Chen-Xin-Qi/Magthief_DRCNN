#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : train.py
@Date         : 2022/04/15 16:38:57
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Train, validate and test the model
'''

from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
from config import CONFIG

class Trainer(object):
    def __init__(self, model, device, optimizer, lr, criterion, num_epochs):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.criterion = criterion
        self.num_epochs = num_epochs
        
    def run(self, train_loader, val_loader, test_loader):
        train_writer = SummaryWriter(log_dir="./logs/train")
        val_writer = SummaryWriter(log_dir="./logs/val")
        min_loss = float("inf")
        for e in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, e)
            val_loss = self.val_epoch(val_loader, e)
            train_writer.add_scalar('Train', train_loss, e)
            val_writer.add_scalar('Val', val_loss, e)
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(self.model.state_dict(), CONFIG["save_model_path"])
                print('Already save model!')
        acc = self.test(test_loader)
        print(f'Test accuracy is: {acc}%')
        
        
    def train_epoch(self, train_loader, e):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0.0
        dataset_size = 0
        for idx, (data, label) in pbar:
            self.optimizer.zero_grad()
            data, label = data.to(self.device), label.to(self.device)
            predict = self.model(data)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()
            total_loss += (loss.item() * CONFIG["BATCH_SIZE"])
            dataset_size += CONFIG["BATCH_SIZE"]
            epoch_loss = total_loss / dataset_size
            pbar.set_description(f"Epoch: {e}  Iter: {idx}  Train Loss: {epoch_loss:.4f}")
            pbar.refresh()
            pbar.update(1)
        return epoch_loss
    
    
    def val_epoch(self, val_loader, e):
        self.model.eval()
        total_loss = 0.0
        dataset_size = 0
        for idx, (data, label) in enumerate(val_loader):
            self.optimizer.zero_grad()
            data, label = data.to(self.device), label.to(self.device)
            predict = self.model(data)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()
            total_loss += (loss.item() * CONFIG["BATCH_SIZE"])
            dataset_size += CONFIG["BATCH_SIZE"]
            epoch_loss = total_loss / dataset_size
        print(f"Epoch: {e} Validate Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    
    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        lambda_val = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                class_predict, _= self.model(x, lambda_val)
                _, predicted = torch.max(class_predict.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = 100 * correct / total
        return acc