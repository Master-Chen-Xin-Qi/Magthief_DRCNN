#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : text.py
@Date         : 2022/05/28 21:53:16
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : simulate the text input action in wechat and qq
'''

import os
import time
import numpy as np

# 26 letters
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
           'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# wechat
# while(1):
#     word_num = np.random.choice(range(1, 7))  # 单词个数
#     print('word_num: {}'.format(word_num))
#     for i in range(word_num):
#         word_len = np.random.choice(range(3, 8))  # 单词长度
#         index = 0
#         while(index<word_len):
#             os.system('adb shell input text {}'.format(np.random.choice(letters)))
#             time.sleep(0.01)
#             index += 1
#         os.system('adb shell input keyevent KEYCODE_SPACE')
#     os.system('adb shell input tap 979 1367')  # 发送
    
# qq    
# while(1):
#     word_num = np.random.choice(range(1, 7))  # 单词个数
#     print('word_num: {}'.format(word_num))
#     for i in range(word_num):
#         word_len = np.random.choice(range(3, 8))  # 单词长度
#         index = 0
#         while(index<word_len):
#             os.system('adb shell input text {}'.format(np.random.choice(letters)))
#             time.sleep(0.01)
#             index += 1
#         os.system('adb shell input keyevent KEYCODE_SPACE')
#     os.system('adb shell input tap 964 1274')  # 发送
    
# word
while(1):
    word_num = np.random.choice(range(1, 7))  # 单词个数
    print('word_num: {}'.format(word_num))
    for i in range(word_num):
        word_len = np.random.choice(range(3, 8))  # 单词长度
        index = 0
        while(index<word_len):
            os.system('adb shell input text {}'.format(np.random.choice(letters)))
            time.sleep(0.01)
            index += 1
        os.system('adb shell input keyevent KEYCODE_SPACE')
        if(np.random.rand() < 0.2):
            os.system('adb shell input keyevent KEYCODE_DEL')  # 一定概率删除