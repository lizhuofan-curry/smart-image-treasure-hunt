# 本文件负责集中保存商品分类模块的数据路径、类别信息和训练超参数

from pathlib import Path

# ===================1.路径配置=======================

# 当前文件位于项目根目录/image_classification/classification_config.py
# 所以项目根目录为
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 当前分类模块目录
CLASSIFICATION_DIR = Path(__file__).resolve().parent

# common 公共资源目录
COMMON_DIR  = PROJECT_ROOT / 'common'

# 商品图片目录
IMG_DIR = COMMON_DIR / 'dataset'

# 商品分类标签文件
FASHION_LABELS_PATH = COMMON_DIR / 'fashion-labels.csv'

# ===================2.图片配置====================

# 分类模型输入图片的高度和宽度
IMG_H = 64
IMG_W = 64

# ==================3.数据集划分配置====================

# 随机种子，用于固定数据集划分和训练随机性
SEED = 42

# 训练集占完整数据集的比例
TRAIN_RATIO = 0.75

# 验证集占完整数据集的比例
VAL_RATIO = 1-TRAIN_RATIO

# ====================4.训练超参数======================

# 优化器学习率
LEARNING_RATE = 1e-3

# 训练阶段每个 batch 的图片数量
TRAIN_BATCH_SIZE = 32

# 验证阶段每个 batch 的图片数量
VAL_BATCH_SIZE = 32

# 测试阶段每个 batch 的图片数量
TEST_BATCH_SIZE = 32

# 模拟训练轮数
EPOCHS = 10

# Windows 环境先使用 0，避免多进程加载报错
NUM_WORKERS = 0

# 使用 CUDA 时，加快 CPU 到 GPU 的数据传输
PIN_MEMORY = True

# =======================5.类别配置=======================

# 商品类别总数
NUM_CLASSES = 5

# 将分类标签编号映射为中文类别名称
CLASSIFICATION_NAMES = {
    0:'上衣',
    1: "鞋",
    2: "包",
    3: "下身衣服",
    4: "手表",
}

# ======================6.模型保存配置=====================

# 分类模块包名
PACKAGE_NAME = 'image_classification'

# 分类模型参数文件名
CLASSIFIER_MODEL_NAME = 'classifier.pth'

# 分类模型参数的完整保存路径
CLASSIFIER_MODEL_PATH = CLASSIFICATION_DIR / CLASSIFIER_MODEL_NAME
