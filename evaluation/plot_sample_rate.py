#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_smple_rate.py
@Date         : 2022/05/22 17:03:12
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot f1 score for different sample rate
'''

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.arange(4)
    sample_rates = [10, 20, 50, 100]
    f1 = [0.154, 0.22, 0.643, 0.925]
    plt.xticks([0, 10, 20, 50, 100])
    plt.ylim(0, 1)
    for i in range(len(sample_rates)):
        plt.bar(x[i], f1[i], width=0.5)
    plt.xticks(x, sample_rates)
    plt.xlabel("Sample rate (Hz)")
    plt.ylabel("Macro F1 score")
    plt.show()
    

    