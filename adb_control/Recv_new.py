# coding:utf-8
'''
代码功能: 接收移动设备的实时磁信号, 50Hz only
'''
import socket, time, os
import pandas as pd, numpy as np
import math

saveTime = 1  # Minutes
train_folder = './raw_mag_data/'

def collect_mag(data_save):
    folder_save = train_folder
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    if os.path.exists(folder_save + data_save):
        os.remove(folder_save + data_save)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("192.168.1.102", 8090))  # ip address
    packages_data = []
    save_packages = 500
    st = time.time()
    count_packages = 0
    while True:
        # 读数
        data, _ = sock.recvfrom(150)
        # print("111")
        one_package = data.decode('ascii')
        one_package = one_package.split(',')
        if len(one_package) >= 9:  # WARNING HERE: Different phones are different
            # print(one_package)
            # one_package = float(one_package[-3])
            one_package = [float(one_package[-3]), float(one_package[-2]), float(one_package[-1]), float(one_package[0])]
            packages_data.append(one_package)  # Four info: x-mag,y-mag,z-mag,time
            count_packages += 1
        # 存盘
        if len(packages_data) > save_packages:
            fid = open(folder_save + data_save, 'a')
            packages_data = pd.DataFrame(packages_data)
            packages_data.to_csv(fid, header=False, index=False)
            fid.close()
            packages_data = []
            print('Saving mag data\t', count_packages)
        et = time.time()  
        if et - st > saveTime * 60:
            print('Mag Sensor Used time: %d seconds, collected %d samples' % (et - st, count_packages))
            break

if __name__ == '__main__':

    # Collect data
    data_save = 'huawei.txt'
    collect_mag(data_save)