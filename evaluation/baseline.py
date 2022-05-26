#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : baseline.py
@Date         : 2022/05/22 16:44:15
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Baseline comparison between DRCNN, SVM, Random Forest, and CNN
'''
import sys
import os
sys.path.append(os.getcwd())  # add current path
print(sys.path)
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from dataset import get_pic_and_labels, get_data
from config import CONFIG

if __name__ == '__main__':
    pic_names, labels = get_pic_and_labels(CONFIG["pics_path"])
    train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data(pic_names, labels)
    
    # SVM model
    svm_model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
    svm_model.fit(train_data, train_labels)
    svm_pred_label = svm_model.predict(test_data)
    svm_macro_f1 = f1_score(test_labels, svm_pred_label, average='macro')  # macro-average
    
    # Random Forest
    tree = RandomForestClassifier(n_estimators=100)
    tree.fit(train_data, train_labels)
    rf_pred_label = tree.predict(test_data)
    rf_macro_f1 = f1_score(test_labels, rf_pred_label, average='macro')  # macro-average
    
    print(f"F1 score, SVM: {svm_macro_f1:.4f}, Random Forest: {rf_macro_f1:.4f}")
    
    
    
    
    