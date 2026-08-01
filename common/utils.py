# 工具函数
import os
import random
import numpy as np
import torch


# 对所有库设置相同的随机数种子，保证训练过程尽量可复现
def seed_everything(seed=42):
    random.seed(seed)  # Python 内置随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置哈希种子
    np.random.seed(seed)  # NumPy 随机数种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数种子

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前 GPU 随机数种子
        torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机数种子

    # 主要管 cuDNN 卷积相关操作
    torch.backends.cudnn.deterministic = True  # 保证 CuDNN 操作确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动选择优化算法

# 正则表达式模块
import re

# 将文件名的列表，按照字母和数字进行排序
def sorted_alphanum(file_names):
    # 定义转换函数：将数字转为int，非数字转为小写形式
    convert = lambda str: int(str) if str.isdigit() else str.lower()
    # 获取排序键函数：列表表达式，将原文件名切分开，并转换
    alphanum_key = lambda name: [ convert(str) for str in re.split('([0-9]+)', name) ]
    # 将原文件名列表，按键排序，并返回
    return sorted(file_names, key=alphanum_key)


