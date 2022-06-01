# README

## I. 数据集

- 本项目数据集均采集自Redmi K30 pro，采样率100Hz，共包括15种app
- 每种app内部操作数据时长为10-15分钟左右
- 时间窗口滑动率设为0.15，以保证数据的连续相关性

## II. 代码结构

```python
.
|-- DRCNN_modules  # DRCNN相关函数
|-- STN.py  # STN模型
|-- adb_control  # adb控制手机采集数据脚本
|-- config.py  # 参数设置
|-- dataset.py  # 自制数据集类
|-- drcnn_labels.npy  
|-- evaluation 
|-- figs  #保存spectrum
|-- gt_figs  # 保存得到的ground truth
|-- logs  # tensorboard可视化保存路径
|   |-- STN_train
|   |-- STN_val
|   |-- train
|   `-- val
|-- mag_np  # 保存npy格式的数据
|-- main.py  # DRCNN主程序
|-- STN_main.py  # STN主程序
|-- name_to_label.txt  # 名字和标签转换
|-- raw_mag_data  # 源数据
|-- requirements.txt  # 依赖库
|-- save_checkpoints  # 保存训练节点
|   |-- DRCNN
|   `-- STN
|-- train.py  # 训练部分
`-- utils.py  # 项目中用到的函数
```

## III. 训练方法

1. 获取项目所需所有模块

   ```bash
   pip install -r requirments.txt
   ```

2. 生成数据的spectrum，图片将被保留在figs文件夹下

   ```bash
   python utils.py
   ```

3. 训练STN，详见`STN_main.py`中的Step 1，会在`gt_fig`下生成ground truth区域图片 

   ```bash
   python STN_main.py
   ```

4. 训练DRCNN，详见`main.py`中的Step 2

   ```bash
   python main.py
   ```

