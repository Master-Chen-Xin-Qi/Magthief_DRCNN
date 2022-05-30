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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from config import CONFIG
import DRCNN_modules.array_tool as at

from collections import namedtuple
LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

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
        
    def train_epoch(self, train_loader, e):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0.0
        dataset_size = 0
        for idx, (img, bbx, label) in pbar:
            img, bbx, label = img.to(self.device), bbx.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            losses = self.forward(img, bbx, label)
            losses.total_loss.backward()
            self.optimizer.step()
            total_loss += (losses.total_loss.item() * CONFIG["BATCH_SIZE"])
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
        for idx, (img, bbx, label, scale) in enumerate(val_loader):
            self.optimizer.zero_grad()
            img, bbx, label = img.to(self.device), bbx.to(self.device), label.to(self.device)
            losses = self.forward(img, bbx, label, scale)
            losses.total_loss.backward()
            self.optimizer.step()
            total_loss += (losses.total_loss.item() * CONFIG["BATCH_SIZE"])
            dataset_size += CONFIG["BATCH_SIZE"]
            epoch_loss = total_loss / dataset_size
        print(f"Epoch: {e} Validate Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def forward(self, imgs, bboxes, labels):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.model.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.model.rpn(features, img_size, scale=1)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
    
class STN_Trainer(object):
    """
    trainer for STN, run to generate the gt boxes for RPN, if there are multiple apps, 
    we need to use the first app's STN model to generate the first gt box, and use
    the scond app's STN model to generate the second gt box.
    """
    def __init__(self, model, device, optimizer, lr, criterion, num_epochs):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.criterion = criterion
        self.num_epochs = num_epochs
        
    def run(self, train_loader, val_loader, pos_app):
        train_writer = SummaryWriter(log_dir=f"./logs/STN_train/{pos_app}")
        val_writer = SummaryWriter(log_dir=f"./logs/STN_val/{pos_app}")
        min_loss = float("inf")
        for e in range(1, self.num_epochs+1):
            train_loss = self.train_epoch(train_loader, e)
            val_loss = self.val_epoch(val_loader, e)
            train_writer.add_scalar('Train', train_loss, e)
            val_writer.add_scalar('Val', val_loss, e)
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(self.model.state_dict(), CONFIG["save_STN_path"]+'_'+pos_app+'.pt')
                print('Already save model!')
        
    def train_epoch(self, train_loader, e):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0.0
        dataset_size = 0
        for idx, (data, label) in pbar:
            self.optimizer.zero_grad()
            
            # show the figure
            # from PIL import Image
            # import torchvision.transforms as transforms
            # pics = transforms.ToPILImage()(data.squeeze(0)[0, :, :])
            # pics.save('data.png')
            # img = Image.open('data.png')
            # img.show()
            
            data, label = data.to(self.device), label.to(self.device)
            predict = self.model(data)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()
            total_loss += (loss.item() * CONFIG["BATCH_SIZE"])
            dataset_size += CONFIG["BATCH_SIZE"]
            epoch_loss = total_loss / dataset_size
            pbar.set_description(f"Epoch: {e}  Iter: {idx}  Train Loss: {epoch_loss:.4f}")
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
        print(f"Epoch: {e}  Validate Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def test(self, test_loader, pos_app):
        self.model.eval()
        self.model.load_state_dict(torch.load(CONFIG["save_STN_path"]+'_'+pos_app+'.pt'))
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                class_predict = self.model(x)
                predicted = torch.max(class_predict.data, 1)[1]
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = 100 * correct / total
        return acc
    
    def generate_gt_boxes(self, loader, pos_app):
        '''
        generate the gt boxes for positive labels
        '''
        self.model.eval()
        self.model.load_state_dict(torch.load(CONFIG["save_STN_path"]+'_'+pos_app+'.pt'))
        
        # save gt box figs
        # pics = []
        # with torch.no_grad():
        #     for _, (data, label) in enumerate(loader):
        #         data, label = data.to(self.device), label.to(self.device)
        #         pic, = self.model.generate_box(data)
        #         pics.append(pic)
        # return pics    

        # save gt box y1, y2 and label
        labels = []
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                data = data.to(self.device)
                y1, y2 = self.model.generate_box(data)
                labels.append([(y1, y2), label.item()])
        return labels
     
     
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()           
                
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss