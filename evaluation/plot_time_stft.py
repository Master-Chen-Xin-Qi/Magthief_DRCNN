#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_time_stft.py
@Date         : 2022/05/30 09:30:56
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot figure of both time series and stft
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.load('mag_np/all_data.npy')
min_max_scalar = MinMaxScaler()
min_max_data = min_max_scalar.fit_transform(data)
plt.subplot(211)
x = np.arange(0, 5, 1/100)
plt.plot(x, min_max_data[0, :])
plt.xlim(0, 5)
plt.ylabel('Amplitude')
plt.subplot(212)
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(min_max_data[0, :], NFFT=32, Fs=100, noverlap=30)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()
