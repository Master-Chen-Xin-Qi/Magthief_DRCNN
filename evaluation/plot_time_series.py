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
    
    aiqiyi_data = min_max_data[:aiqiyi_len, :].reshape(-1, 1)
    baidu_data = min_max_data[aiqiyi_len:aiqiyi_len+baidu_len, :].reshape(-1, 1)
    netmusic_data = min_max_data[aiqiyi_len+baidu_len:aiqiyi_len+baidu_len+netmusic_len, :].reshape(-1, 1)
    
    plt.subplot(311)
    plt.plot(aiqiyi_data)
    plt.ylabel('aiqiyi')
    plt.subplot(312)
    plt.plot(baidu_data)
    plt.ylabel('baidu')
    plt.subplot(313)
    plt.plot(netmusic_data)
    plt.ylabel('netmusic')
    plt.show()
