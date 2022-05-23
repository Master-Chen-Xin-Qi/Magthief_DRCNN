#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_baseline.py
@Date         : 2022/05/23 13:52:50
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot f1 score for different baseline and DRCNN
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    baselines = ["SVM", "RF", "CNN", "DRCNN"]
    f1_inapp = [0.55, 0.62, 0.743, 0.925]  # in-app, need to change the value
    f1= [0.523, 0.589, 0.704, 0.85]  # app
    plt.xticks([0, 10, 20, 50, 100])
    plt.ylim(0, 1)
    
    '''多数据并列柱状图'''

    x=np.arange(4)

    bar_width=0.35

    plt.bar(x, f1_inapp,bar_width,color="g",align="center",label="in-app",alpha=0.5)
    plt.bar(x+bar_width, f1,bar_width,color="r",align="center",label="app",alpha=0.5)

    plt.ylabel("Macro F1 score")

    plt.xticks(x+bar_width/2, baselines)
    plt.legend()
    plt.show()
    