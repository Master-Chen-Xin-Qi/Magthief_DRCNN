#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : slide.py
@Date         : 2022/05/28 19:29:42
@Author       : Xinqi Chen 
@Software     : VScode.
@Description  : simulate the finger slide action
'''

import os
import time
import numpy as np

x1, y1, x2, y2, h = 565,1527,550,662,50
start = time.time()
while(1):
    os.system('adb shell input swipe {} {} {} {} {}'.format(x1, y1, x2, y2, h))
    watch_time = np.random.choice(range(10, 30))
    time.sleep(watch_time)
    print('Watching for {} seconds'.format(watch_time))
    if(time.time()-start > 15*60):
        print('Already record 15 miniutes, exit')