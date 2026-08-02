# 本文件负责读取商品图片，并创建自编码器所需的完整数据集，训练集和验证集

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T

from common.utils import sorted_alphanum

from image_similarity.similarity_config import (
    IMG_DIR,
    IMG_H,
    IMG_W,
    TRAIN_RATIO,
    SEED
)

# 获取商品图片目录中的全部图片路径
def get_image_paths(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f'没有找到商品图片目录：{image_dir}'
        )
    image_paths = []

    # 支持的数据集图片格式
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    for pattern in image_patterns:
        current_paths = list(image_dir.glob(pattern))
        image_paths.extend(current_paths)

    if len(image_paths) == 0:
        raise RuntimeError(
            f'在{image_dir}中没有找到商品图片'
        )

    # 自然排序保证 2.jpg 排在 10.jpg 前面
    # 后面 embedding 的第 i 行将对应这里的第 i 张图片
    image_paths = sorted_alphanum([str(path) for path in image_paths])
    image_paths = [Path(path) for path in image_paths]

    return image_paths

'''
商品图片重建数据集
返回：
    input_image:
        输入自编码器的图片
        形状为 [3,IMG_H,IMG_W]
    target_image:
        自编码器需要重建的目标图片
        形状为 [3,IMG_H,IMG_W]    
'''
class ImageReconstructionDataset(Dataset):
    def __init__(
            self,
            image_dir,
            transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform

        # 保存经过自然排序的全部图片路径
        self.image_paths = get_image_paths(image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 根据索引取得对应图片路径
        image_path = self.image_paths[index]

        # 打开图片并统一转换成 RGB 三通道
        with Image.open(image_path) as image:
            image = image.convert('RGB')

            if self.transform is None:
                raise ValueError('transform 不能为 None')
            tensor_image = self.transform(image)

        # 普通自编码器的输入和目标是同一张干净图片
        # 这里是同一张图片扮演两个角色
        # 第一个tensor:模型输入 第二个tensor : 模型应该重建出的目标
        return tensor_image,tensor_image

# 新设置一个transform函数是为了保证训练图片的预处理和外部查询图片的预处理是同一套
def create_transform():
    transform = T.Compose([
        T.Resize((64, 48)),
        T.Pad(
            padding=(8, 0, 8, 0),
            fill=255,
            padding_mode='constant',
        ),
        T.ToTensor(),
    ])
    return transform

# 创建完整商品图片数据集
# 并按固定随机种子划分训练集和验证集
# 返回完整数据集，训练集和验证集
# train:用于训练编码器和解码器
# val : 用于观察模型重建新图片的能力
# full :训练结束后生成全部商品的 embedding.npy
def create_dataset():

    transform = create_transform()

    # 包含全部商品图片的数据集
    full_dataset = ImageReconstructionDataset(
        image_dir=IMG_DIR,
        transform=transform
    )

    # 计算训练集样本数量
    train_size = int(TRAIN_RATIO * len(full_dataset))
    # 计算验证集样本数量
    val_size = len(full_dataset) - train_size

    # 固定数据划分结果
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    return full_dataset, train_dataset, val_dataset

if __name__ == "__main__":
    full_dataset, train_dataset, val_dataset = (
        create_dataset()
    )

    print(
        "完整数据集样本数量：",
        len(full_dataset),
    )

    print(
        "训练集样本数量：",
        len(train_dataset),
    )

    print(
        "验证集样本数量：",
        len(val_dataset),
    )

    input_image, target_image = (
        train_dataset[0]
    )

    print(
        "输入图片形状：",
        input_image.shape,
    )

    print(
        "目标图片形状：",
        target_image.shape,
    )

    print(
        "输入图片和目标图片是否相同：",
        torch.equal(
            input_image,
            target_image,
        ),
    )

    print(
        "第一张完整数据集图片：",
        full_dataset.image_paths[0].name,
    )