#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_time_series.py
@Date         : 2022/05/30 17:18:43
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot the difference between different data
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    data = np.load('mag_np/all_data.npy')
    min_max_scalar = MinMaxScaler()
    min_max_data = min_max_scalar.fit_transform(data)
    aiqiyi_len = len(np.load('mag_np/aiqiyi_500.npy'))
    baidu_len = len(np.load('mag_np/baidu_500.npy'))
    netmusic_len = len(np.load('mag_np/netmusic_500.npy'))
    
    plot_len = 10
    aiqiyi_data = min_max_data[:plot_len, :].reshape(-1, 1)
    baidu_data = min_max_data[aiqiyi_len:aiqiyi_len+plot_len, :].reshape(-1, 1)
    netmusic_data = min_max_data[aiqiyi_len+baidu_len:aiqiyi_len+baidu_len+plot_len, :].reshape(-1, 1)
    
    x = np.arange(0, plot_len*5, 1/100)
    plt.subplot(311)
    plt.plot(x, aiqiyi_data)
    plt.ylim(0, 1)
    plt.ylabel('Aiqiyi')
    plt.subplot(312)
    plt.plot(x, baidu_data)
    plt.ylim(0, 1)
    plt.ylabel('Baidu')
    plt.subplot(313)
    plt.plot(x, netmusic_data)
    plt.ylim(0, 1)
    plt.ylabel('Netmusic')
    plt.xlabel('Time (s)')
    plt.show()
