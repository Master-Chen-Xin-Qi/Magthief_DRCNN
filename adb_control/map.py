#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : map.py
@Date         : 2022/05/29 15:55:39
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : collect data in baidu map and gaode map
'''

import os
import time

start = time.time()
real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('Start time: {}'.format(real_time))
while(1):
    os.system('adb shell input swipe 865 1527 650 1500 50')
    time.sleep(1)
    # os.system('adb shell input tap 964 1274')
    time.sleep(1)
    os.system('adb shell input swipe 865 1527 650 1500 50')
    if(time.time()-start > 15*60):
        print('Already record 15 miniutes, exit')