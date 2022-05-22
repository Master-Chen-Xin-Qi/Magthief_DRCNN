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

if __name__ == '__main__':
    sample_rates = [10, 20, 50, 100]
    f1 = [0.154, 0.22, 0.643, 0.925]
    plt.plot(sample_rates, )
    plt.xlabel("Sample rate")
    plt.ylabel("Macro F1 score")
    plt.show()
    