# 负责读取商品图片，并完成图像去噪模型所需的数据预处理
import random

from image_denoising.denoising_config import (
    IMG_PATH,
    TRAIN_SALT_PEPPER_MIN,
    TRAIN_SALT_PEPPER_MAX,
    TRAIN_GAUSSIAN_NOISE_MIN,
    TRAIN_GAUSSIAN_NOISE_MAX,
    TRAIN_RATIO,
    SEED
)
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,Subset
from pathlib import Path

from PIL import Image
from torchvision import transforms as T

# 复用 common 中已经写好的自然排序函数
from common.utils import sorted_alphanum

# 获取全部图片路径
'''
获取指定目录中的所有商品图片路径
参数：
    image_dir: 商品图片文件夹路径
返回：
    image_paths: 排序后的图片完整路径列表    
'''
def get_image_paths(image_dir):

    # glob 不会因为目录不存在而直接报错，而是返回空列表
    if not image_dir.exists():
        raise FileNotFoundError(
            f'没有找到数据集目录：{image_dir}'
        )
    image_paths = []

    # dataset 中可能不止有 jpg,因此分别查找几种常见格式
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.webp']
    for pattern in image_patterns:
        # 找到该目录下所有的 pattern 图片
        current_paths = list(image_dir.glob(pattern))
        image_paths.extend(current_paths)
    if len(image_paths) == 0:
        raise RuntimeError(f'在{image_dir}中没有找到图片')

    # common 中的 sorted_alphanum 主要按照 字符串处理文件名
    # 因此先把path 转成字符串自然排序
    image_paths = sorted_alphanum([str(path)for path in image_paths])

    # 排序后重新转换成 Path对象
    # 后续仍然可以继续用 Path 的各种操作
    image_paths = [Path(path) for path in image_paths]
    return image_paths


'''
给图像添加椒盐噪声
参数
    image 
        [3,H,W] 的清晰照片 Tensor
    
    noise_ratio:
        被替换成黑点或者白点的像素比例
        例如 0.03 表示大约 3% 的像素受到影响
返回：
    添加椒盐噪声后的图片        
'''
def add_salt_pepper_noise(image,
                          noise_ratio,
                          generator = None):    # 训练集传 None,噪声每次随机变化，验证集传固定生成器
    noisy_image = image.clone() # clone 创建一份图片副本，避免直接修改作为训练目标的 clean_image

    # 只创建一张空间随机图
    # 形状为[1，H，W]
    random_map = torch.rand(
        1,
        image.shape[1],
        image.shape[2],
        generator=generator,
        device=image.device
    )
    pepper_mask = random_map < noise_ratio/2
    salt_mask = random_map > 1 - noise_ratio/2

    # mask 会广播到 RGB 三个通道
    noisy_image = torch.where(
        pepper_mask,
        torch.zeros_like(noisy_image),
        noisy_image
    )
    # torch.where 相当于 ? : 三元运算符
    # 满足条件用第一个值，不满足用第二个值
    # 用布尔掩码的化需要把通道扩到三维
    noisy_image = torch.where(
        salt_mask,
        torch.ones_like(noisy_image),
        noisy_image
    )

    return noisy_image

# 噪声函数不在绑定固定配置，增加随机性，提高泛化能力
def add_gaussian_noise(image,
                       noise_factor,
                       generator= None):
    gaussian_noise = torch.randn( size=image.shape,
                                  generator=generator,
                                  device=image.device,
                                  dtype=image.dtype)
    noisy_image = image+noise_factor*gaussian_noise
    # 由于randn可能随机产数负数，添加高斯噪声后部分像素可能超出 [0,1]
    # 用clamp把它们限制在[0,1] 大于1的为1，小于0的为0
    noisy_image = torch.clamp(noisy_image, min=0.0, max=1.0)
    return noisy_image

# 自定义dataset 类
class ImageDenoisingDataset(Dataset):

    def __init__(
            self,
            image_dir,
            transform= None,
            gaussian_noise_min = TRAIN_GAUSSIAN_NOISE_MIN,
            gaussian_noise_max = TRAIN_GAUSSIAN_NOISE_MAX,
            salt_pepper_min = TRAIN_SALT_PEPPER_MIN,
            salt_pepper_max = TRAIN_SALT_PEPPER_MAX,
            is_train = True,
            noise_seed = SEED
    ):
        self.image_dir = image_dir  # 保存图片所在目录

        self.transform = transform  # 保存图片预处理操作

        # 保存高斯噪声的随机范围
        self.gaussian_noise_min = gaussian_noise_min
        self.gaussian_noise_max = gaussian_noise_max

        # 保存椒盐噪声的随机范围
        self.salt_pepper_min = salt_pepper_min
        self.salt_pepper_max = salt_pepper_max

        #  True 表示是训练集，噪声每次随机变化
        # False 表示验证集，噪声固定
        self.is_train = is_train

        # 验证集生成固定噪声时使用
        self.noise_seed = noise_seed

        self.image_paths = get_image_paths(image_dir) # 保存图片路径

    def __len__(self):
        return len(self.image_paths) # 返回数据集中图片总数

    def __getitem__(self, index):
        # 根据 index 找到一张图片路径
        image_path = self.image_paths[index]

        # 打开图片，转换成 RGB 三通道
        clean_image = Image.open(image_path).convert('RGB')

        # 执行 Resize 和 ToTensor
        if self.transform is  None:
            raise ValueError('transform 不能为None')

        clean_image = self.transform(clean_image)   # 将它放缩和转成tensor

        # 创建随机数生成器
        if self.is_train:
            # 训练集：每次读取图片时产生不同的随机噪声
            python_rng = random
            torch_generator = None
        else:
            # 验证集：同一个 index 每次使用相同的种子，因此每轮验证获得相同噪声
            current_seed = (self.noise_seed + index)
            python_rng = random.Random(current_seed)
            torch_generator = torch.Generator().manual_seed(current_seed)

        # 在[0.05,0.25]之间随机生成一个高斯噪声的强度
        gaussian_noise_factor = python_rng.uniform(
            self.gaussian_noise_min,
            self.gaussian_noise_max
        )

        # 在 [0.01，0.04]之间生成椒盐比例的随机数
        salt_pepper_ratio = python_rng.uniform(
            self.salt_pepper_min,
            self.salt_pepper_max
        )

        # 随机产生 0，1，2 中的一个整数，用于决定本次采用哪种噪声
        # 让训练时的混合噪声居多
        noise_type = python_rng.choices(
            population= [0,1,2],    # 表示可以选择的结果
            weights = [0.3,0.3,0.4], # 表示三个结果的相对概率
            k = 1,                   # 表示只随机选择一个结果
        )[0]  # 因为这个返回的是一个列表，列表中只有一个值，所以取第0个

        if noise_type == 0:
            # 情况1：只添加高斯噪声
            noisy_image = add_gaussian_noise(
                image=clean_image,
                noise_factor = gaussian_noise_factor,
                generator= torch_generator
            )

        elif noise_type == 1:
            # 只添加椒盐噪声
            noisy_image = add_salt_pepper_noise(
                image=clean_image,
                noise_ratio=salt_pepper_ratio,
                generator=torch_generator
            )
        else:
            # 先添加高斯噪声
            noisy_image = add_gaussian_noise(
                image=clean_image,
                noise_factor = gaussian_noise_factor,
                generator=torch_generator
            )
            # 再叠加椒盐噪声
            noisy_image = add_salt_pepper_noise(
                image = noisy_image,
                noise_ratio=salt_pepper_ratio,
                generator=torch_generator
            )

        return noisy_image,clean_image

def create_transform():
    # 创建训练、测试和网页推理共用的图片预处理

    return T.Compose([
        # 保持原商品图大致比例
        # 高度缩放为 64，宽度缩放为 48
        T.Resize((64, 48)),

        # 左右各补 8 个白色像素
        # 最终得到 64×64
        T.Pad(
            padding=(8, 0, 8, 0),
            fill=255,
            padding_mode="constant",
        ),

        # PIL 图片转换为 [3, 64, 64] Tensor
        T.ToTensor(),
    ])

# 创建完整去噪数据集，并划分训练集和验证集
def create_dataset():
    # 图片预处理只在创建数据集时统一定义
    # 不能强制拉伸成正方形 60 x 80 -> 64 x 64 这样人物会被横向拉宽
    transform = create_transform()

    # 创建训练数据源
    # 训练集噪声每次随机变化
    train_source = ImageDenoisingDataset(
        image_dir=IMG_PATH,
        transform=transform,
        is_train= True,
        noise_seed = SEED
    )

    # 创建验证数据源
    # 验证集噪声根据图片 index 固定
    val_source = ImageDenoisingDataset(
        image_dir=IMG_PATH,
        transform= transform,
        is_train= False,
        noise_seed = SEED + 10000,
    )

    # 生成固定划分索引
    dataset_size = len(train_source)

    # 根据配置比例计算两个子集的样本数量
    train_size = int(dataset_size * TRAIN_RATIO)
    val_size = dataset_size - train_size

    # 固定本次划分使用的随机种子

    generator = torch.Generator().manual_seed(SEED)

    # 生成 0 ~ dataset_size - 1 的随机排列
    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    train_indices = (shuffled_indices[:train_size])
    val_indices = (shuffled_indices[train_size:])

    # 创建训练集和验证集
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    return train_dataset,val_dataset


# 当前阶段测试
if __name__ == '__main__':

    train_dataset,val_dataset = create_dataset()

    print("数据集目录：", IMG_PATH)
    print("训练集图片数量：", len(train_dataset))
    print("验证集图片数量：", len(val_dataset))

    train_noisy_1, _ = train_dataset[0]
    train_noisy_2, _ = train_dataset[0]

    print(
        "训练集两次噪声是否相同：",
        torch.equal(
            train_noisy_1,
            train_noisy_2,
        )
    )

    valid_noisy_1, _ = val_dataset[0]
    valid_noisy_2, _ = val_dataset[0]
    print('验证集两次噪声是否相同：',torch.equal(valid_noisy_1,valid_noisy_2))
    noisy_image, clean_image = train_dataset[0]

    # Tensor 图片的形状是 [通道, 高度, 宽度]
    # matplotlib 显示图片时需要 [高度, 宽度, 通道]
    # 因此使用 permute 将维度从 [C, H, W] 改为 [H, W, C]
    clean_image_show = clean_image.permute(1, 2, 0).numpy()
    noisy_image_show = noisy_image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))

    # 显示清晰图片
    plt.subplot(1, 2, 1)
    plt.imshow(clean_image_show)
    plt.title("Clean Image")
    plt.axis("off")

    # 显示带噪图片
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image_show)
    plt.title("Noisy Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

