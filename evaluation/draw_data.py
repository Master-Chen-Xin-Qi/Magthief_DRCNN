#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : draw_data.py
@Date         : 2022/05/16 21:00:10
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Draw the weibo data, typing text and watching video
'''

import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    mag_data = []
    fid = open(file_name, 'r')
    for line in fid:
        line = line.strip('\n')
        data = line.split(',')
        data_x = float(data[0])
        mag_data.append(data_x)
    return np.array(mag_data)


if __name__ == '__main__':
    work_data_name = '../work1_weibo_zhijia.txt'
    video_data_name = '../video1_weibo_zhijia.txt'
    work_data = read_data(work_data_name)[:3000]
    video_data = read_data(video_data_name)[3000:6000]
    
    X = np.hstack((work_data, video_data))
    X_min = np.min(X)
    X_max = np.max(X)
    for i in range(len(X)):
        X[i] = (X[i] - X_min) / (X_max - X_min)

    work = X[:3000]
    video = X[3000:6000]
    x = np.arange(0, 30, 0.01)
    ax1 = plt.subplot(211)
    plt.plot(x, work)
    # plt.xlabel('time (s)')
    plt.ylabel('mag')
    ax2 = plt.subplot(212)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(work, NFFT=128, Fs=100, noverlap=64)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.savefig("./1.pdf")
    plt.close()
    
    ax3 = plt.subplot(211)
    plt.plot(x, video)
    # plt.xlabel('time (s)')
    plt.ylabel('mag')
    ax4 = plt.subplot(212)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(video, NFFT=128, Fs=100, noverlap=64)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.savefig("./2.pdf")
    plt.show()