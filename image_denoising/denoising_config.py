# 本文件负责集中管理图像去噪模块的数据、噪声和训练配置

from pathlib import Path

# 1.项目路径配置

'''
__file__ 表示当前文件 denoising_data.py的位置
当前文件 ： 智图寻宝项目/image_denoising/denoising_data.py
.resolve() 将他转换成完整的绝对路径

第一个 parent 智图寻宝项目/image_denoising
第二个 parent
'''
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 商品图片数据集目录
IMG_PATH = PROJECT_ROOT / 'common' / 'dataset'

# 2. 图片预处理配置
IMG_H = 64
IMG_W = 64

# 3. 随机性与数据集划分

# 固定随机种子，让训练集与验证集的划分结果尽量保持一致
SEED = 42

# 75% 图片用于训练
TRAIN_RATIO = 0.75

# 剩下 25% 图片用于验证
VAL_RATIO = 1-TRAIN_RATIO

# 4.噪声配置

# 训练阶段高斯噪声的随机强度范围
TRAIN_GAUSSIAN_NOISE_MIN = 0.05
TRAIN_GAUSSIAN_NOISE_MAX = 0.20

# 训练阶段椒盐噪声的随机比较范围
TRAIN_SALT_PEPPER_MIN = 0.005
TRAIN_SALT_PEPPER_MAX = 0.02

# 测试或者网页演示阶段使用的固定噪声强度
TEST_GAUSSIAN_NOISE_FACTOR = 0.12
TEST_SALT_PEPPER_RATIO = 0.01

# 5.训练超参数

# 学习率
LEARNING_RATE = 1e-4

# 每个训练 batch 包含的图片数量
TRAIN_BATCH_SIZE = 32

# 每个验证 batch 包含的图片数量
VAL_BATCH_SIZE = 32

# 测试阶段每个 batch 包含的图片数量
TEST_BATCH_SIZE = 32

# 模拟训练轮数
EPOCHS = 30

# 6. 模型保存配置

# 当前模块的报名，后续web 接口加载模型时可能使用
PACKAGE_NAME = "image_denoising"

# 去噪模型参数的文件名
DENOISER_MODEL_NAME = "denoiser.pth"

# 去噪模型参数的完整保存路径
DENOISER_MODEL_PATH = PROJECT_ROOT / 'image_denoising'/ DENOISER_MODEL_NAME


