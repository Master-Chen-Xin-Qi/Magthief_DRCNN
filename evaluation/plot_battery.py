#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_battery.py
@Date         : 2022/05/28 11:42:03
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot the result of different battery level
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(3)
y = [0.78, 0.856, 0.92]
battery = ['0-50%', '50%-100%', 'Charging']
plt.ylim(0, 1)
plt.xticks(x, battery)
plt.xlabel('Battery level')
plt.ylabel("Macro F1 score")
plt.bar(x, y, width=0.3)
plt.show()
