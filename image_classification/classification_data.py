# 本文件负责读取商品图片及其分类标签，并创建训练集和验证集

from pathlib import Path

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T

from common.utils import sorted_alphanum
from image_classification.classification_config import (
    IMG_DIR,                # 图片路径
    FASHION_LABELS_PATH,    # 标签路径
    IMG_H,                  # 处理后图片的高
    IMG_W,                  # 处理后图片的宽
    TRAIN_RATIO,            # 训练集的比例
    SEED,                   # 随机种子
    NUM_CLASSES,            # 类型数量
    CLASSIFICATION_NAMES    # 类型对应的名字
)

def get_image_paths(image_dir):
    # 获取商品图片目录中的全部图片路径

    # 创建图片对象
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f'没有找到商品图片目录：{image_dir}'
        )
    image_paths = []

    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    for pattern in image_patterns:
        # 通过循环来遍历上方的所有后缀名，再通过.glob在图片路径下查找
        current_path = list(image_dir.glob(pattern))
        image_paths.extend(current_path)

    if(len(image_paths)==0):
        raise RuntimeError(
            f'在{image_dir}中没有找到商品图片'
        )

    # 自然排序可以让 2.jpg 排在 10.jpg 前面
    # 先强转为字符串，因为自定义的排序用的是字符串
    image_paths = sorted_alphanum([str(path) for path in image_paths])
    # 再转换为 Path 对象
    image_paths = [Path(path) for path in image_paths]

    return image_paths

def load_labels(label_path):
    # 读取 CSV 标签文件，并建立商品 id 到类别编号的映射
    # 返回的是 label_dict字典 例如 {‘1000’：0}

    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(
            f'没有找到标签文件{label_path}'
        )

    # 因为这个表格是纯数字类型
    # 用pandas.read_csv来读取比较方便
    # 读取后 labels 是一个 DataFrame 类似与一个二维表格
    labels = pd.read_csv(label_path)

    # 定义必须存在的列
    required_columns = {'id','target'}

    # 检查有没有缺少列
    # labels.columns.values大概是array(['id', 'target']
    # 经过 set 变成集合 {'id', 'target'}
    missing_columns = required_columns - set(labels.columns.values)


    if missing_columns:
        raise ValueError(
            '标签文件缺少必要列：'
            f'{missing_columns}'
        )
    labels_dict = {}

    # 遍历 DataFrame 的每一行
    # 每次返回两个内容 当前的索引，当前行的数据
    for _, row in labels.iterrows():

        # id 表示自然排序后图片对应的样本索引
        sample_index = int(row['id'])

        # target 表示类别编号
        target = int(row['target'])

        if target < 0 or target >= NUM_CLASSES:
            raise ValueError(
                f'商品{sample_index }的标签{target}'
                f'超出合法范围 0~{NUM_CLASSES-1}'
            )

        labels_dict[sample_index ] = target
    return labels_dict

class ImageLabelDataset(Dataset):
    # 商品图片分类数据集
    # 返回：
    #    image: [3,IMG_H,IMG_W] 的图片Tensor
    #    label: 商品编号 范围为 0~4
    def __init__(
            self,
            image_dir,
            label_path,
            transform=None,
    ):
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform

        # 获取全部商品图片路径
        self.image_paths = get_image_paths(self.image_dir)

        # 读取商品 id 和类别之间的对应关系
        self.labels_dict = load_labels(self.label_path)

        # 检查图片数量和标签数量是否一致
        if len(self.image_paths) != len(self.labels_dict):
            raise RuntimeError(
                "图片数量与标签数量不一致："
                f"图片数量为 {len(self.image_paths)}，"
                f"标签数量为 {len(self.labels_dict)}"
            )


    def __len__(self):
            return len(self.image_paths)

    def __getitem__(self, index):
            # 根据索引获取图片路径
            image_path = self.image_paths[index]

            # 打开图片并统一转换成 RGB 三通道
            image = Image.open(image_path).convert('RGB')

            # 进行图片转化
            if self.transform is None:
                raise ValueError(
                    'transform 不能为 None'
                )
            image = self.transform(image)

            # 根据图片id 查找真实类别
            # index 表示当前图片在自然排序列表中的位置
            label = self.labels_dict[index]
            return image, label

# 创建transform函数，方便网页上传图片和本地图片统一处理
def create_transform():
    transform = T.Compose([
        # 将原始图片缩放为高 64，宽 48
        T.Resize((64,48)),
        # 左右各补 8 个白色像素，得到 64x64
        T.Pad(
            padding = (8,0,8,0),
            fill = 255,
            padding_mode = 'constant',
        ),
        # PIL 图片转为 [3,64,64] tensor
        T.ToTensor(),
    ])
    return transform

# 创建完整商品分类数据集，并按配置划分训练集和验证集
def create_dataset():
    # 原始照片为 宽 60 高 80
    # 缩放到 高 64 宽 48 ，再左右补白至 64 x 64
    transform = create_transform()

    full_dataset = ImageLabelDataset(
        image_dir=IMG_DIR,
        label_path=FASHION_LABELS_PATH,
        transform=transform,
    )

    # 训练集大小
    train_size = int(len(full_dataset) * TRAIN_RATIO)

    # 验证集大小
    val_size = len(full_dataset) - train_size

    # 固定随机划分
    generator = torch.Generator().manual_seed(SEED)

    train_dataset,val_dataset = random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = create_dataset()

    print(
        "商品图片目录：",
        IMG_DIR,
    )

    print(
        "标签文件：",
        FASHION_LABELS_PATH,
    )

    print(
        "训练集样本数量：",
        len(train_dataset),
    )

    print(
        "验证集样本数量：",
        len(val_dataset),
    )

    image, label = train_dataset[0]

    print(
        "第一张图片形状：",
        image.shape,
    )

    print(
        "第一张图片标签：",
        label,
    )

    print(
        "第一张图片类别：",
        CLASSIFICATION_NAMES[label],)