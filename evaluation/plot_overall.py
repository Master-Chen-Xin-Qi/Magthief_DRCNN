#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_overall.py
@Date         : 2022/05/23 14:46:14
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot the overall f1 score for in-app and app
'''

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    in_app = ["static", "work", "game", "music", "surf", "video"]  # call maybe?
    f1 = [0.9, 0.89, 0.92, 0.79, 0.88, 0.91]
    x = np.arange(6)
    plt.ylim(0, 1)
    for i in range(len(in_app)):
        plt.bar(x[i], f1[i], width=0.5, color='g')
    plt.xticks(x, in_app)
    plt.ylabel("Macro F1 score")
    plt.show()
    
    
    
    app = ["Glory of Kings," "PUBG", "Netease Cloud Music", "QQ Music", "Bilibili", "Aiqiyi", 
           "Tiktok", "Taobao", "Jingdong", "Word", "Baidu map", "Gaode map", "WeChat", "QQ"]
    x = np.arange(len(app))
    f1_app = []
    plt.ylim(0, 1)
    for i in range(len(app)):
        plt.bar(x[i], f1_app[i], width=0.5)
    plt.xticks(x, app)
    plt.ylabel("Macro F1 score")
    plt.show()
    
    