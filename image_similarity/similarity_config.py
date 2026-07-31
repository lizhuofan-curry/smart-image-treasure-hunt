# 本文件负责统一管理相似图片检索模块的数据路径，训练参数和模型文件路径

from pathlib import Path

from image_classification.classification_config import NUM_WORKERS
from image_denoising.denoising_config import TRAIN_RATIO, PACKAGE_NAME

# =================1.项目路径===============

# 当前项目的根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 当前相似图片检索模块目录
SIMILARITY_DIR = Path(__file__).resolve().parent

# 公共资源目录
COMMON_DIR= PROJECT_ROOT / "common"

# 商品图片所在目录
IMG_DIR = COMMON_DIR / "dataset"

# ==================2.图片预处理参数=================

# 模型输入图片高度
IMG_H =64

# 模型输入图片宽度
IMG_W = 64

# ====================3.数据集划分参数===============

# 固定随机种子
SEED = 42

# 训练集占完整数据集的比例
TRAIN_RATIO = 0.75

# 验证集占完整数据集的比例
VAL_RATIO = 1-TRAIN_RATIO

# ======================4.训练参数=====================

# 学习率
LEARNING_RATE = 1e-3

# 训练集批次大小
TRAIN_BATCH_SIZE = 32

# 验证集批次大小
VAL_BATCH_SIZE = 32

# 测试集批次大小
TEST_BATCH_SIZE = 32

# 为全部图片生成特征向量时的批次大小
FULL_BATCH_SIZE = 32

# 模型训练轮数
EPOCHS = 30

# 子进程数量，windows 下先使用 0，减少多进程报错
NUM_WORKERS = 0

# 使用 CUDA 时是否开启锁业内存
PIN_MEMORY = True

# =======================5.相似图片检索参数==================

# 默认返回最相似的 5 张商品图片
NUM_SIMILAR_IMAGES = 5

# =======================6.模型与特征文件路径====================

# 当前模块包名
PACKAGE_NAME = 'image_similarity'

# 编码器,解码器参数文件名称
ENCODER_MODEL_NAME = 'encoder.pth'
DECODER_MODEL_NAME = 'decoder.pth'

# 全部商品图片的特征向量文件名称
EMBEDDING_NAME = 'embeddings.npy'

# 编码器参数完整路径
ENCODER_MODEL_PATH = SIMILARITY_DIR / ENCODER_MODEL_NAME
# 解码器参数完整路径
DECODER_MODEL_PATH = SIMILARITY_DIR / DECODER_MODEL_NAME

# 图片特征向量库完整路径
EMBEDDING_PATH = SIMILARITY_DIR / EMBEDDING_NAME

if __name__ == '__main__':
    print(
        "项目根目录：",
        PROJECT_ROOT,
    )

    print(
        "商品图片目录：",
        IMG_DIR,
    )

    print(
        "编码器保存路径：",
        ENCODER_MODEL_PATH,
    )

    print(
        "解码器保存路径：",
        DECODER_MODEL_PATH,
    )

    print(
        "特征向量库保存路径：",
        EMBEDDING_PATH,
    )

    print(
        "默认返回相似图片数量：",
        NUM_SIMILAR_IMAGES,
    )