#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : start_record.py
@Date         : 2022/05/27 17:16:01
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Cold start and hot start for app
'''

import os 
import time
os.system('adb shell am start com.tencent.mm/com.tencent.mm.ui.LauncherUI')  # 打开微信
print(time.time())
real_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
print(f"Wechat start at Time: {real_time} !")

# os.system('adb shell am start com.example.demo/.MainActivity')  # 打开数据记录demo app
# os.system('adb shell input tap 806 372')  # 点击屏幕位置
# os.system('adb shell input tap 572 789')  # 点击开始记录数据
# start_time = time.time()
# real_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
# print(f"Record start! Time: {real_time}")