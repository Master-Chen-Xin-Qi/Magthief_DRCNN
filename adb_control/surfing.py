#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : surfing.py
@Date         : 2022/05/29 10:04:57
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Surfing in app
'''

import os
import time
import numpy as np

# taobao & jingdong surfing
# while(1):
#     os.system('adb shell input swipe 565 1527 550 1462 10')  # slide
#     time.sleep(2)
#     os.system('adb shell input tap 230 1000')  # touch left area
#     time.sleep(3.5)
#     os.system('adb shell input keyevent BACK')
    
    
# weibo surfing
start = time.time()
while(1):
    os.system('adb shell input swipe 565 1527 550 1462 10')  # slide
    time.sleep(np.random.choice(range(1, 5)))
    os.system('adb shell input tap 530 1000')  # touch left area
    time.sleep(np.random.choice(range(3, 5)))
    os.system('adb shell input keyevent BACK')
    if(time.time()-start > 15*60):
        print('Already record 15 miniutes, exit')
      
