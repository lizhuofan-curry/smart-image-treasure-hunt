# 本文件负责定义相似图片检索模块使用的卷积编码器和卷积解码器

import torch
import torch.nn as nn

from image_similarity.similarity_config import IMG_H, IMG_W

# 编码器中的一个下采样模块
# [batch,in_channels,H,W] -> [batch,out_channels,H/2,W/2]
class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # 提取图片局部特征
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1 ,padding=1,bias=False),
            # 标准化，稳定特征分布
            nn.BatchNorm2d(out_channels),

            # 加入激活函数
            nn.ReLU(inplace=True),

            # 高度和宽度缩小一半
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

    def forward(self, x):
        return self.block(x)

# 卷积编码器
# [batch,3,64,64] -> [batch,256,2,2]
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # [batch,3,64,64] -> [batch,16,32,32]
            EncoderBlock(3,16),

            # [batch,16,32,32] -> [batch,32,16,16]
            EncoderBlock(16,32),

            # [batch,32,16,16] -> [batch,64,8,8]
            EncoderBlock(32,64),

            # [batch,64,8,8] -> [batch,128,4,4]
            EncoderBlock(64,128),

            # [batch,128,4,4] -> [batch,256,2,2]
            EncoderBlock(128,256),
        )

    def forward(self, x):
        return self.encoder(x)

# 解码器中的一个上采样模块
# [batch,in_channels,H,W] -> [batch,out_channels,H*2,w*2]
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # 转置卷积同时改变通道数
            # 并将高度和宽度放大两倍 todo
            # 转置卷积的输出尺寸公式为 (输入尺寸 - 1) × stride - 2 × padding+ kernel_size
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride = 2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# 卷积解码器
# [batch,256,2,2] -> [batch,3,64,64]
class ConvDecoder(nn.Module):
    def __init__(self,):
        super().__init__()

        self.decoder  = nn.Sequential(
            # [batch,256,2,2] -> [batch,128,4,4]
            DecoderBlock(256,128),

            # [batch,128,4,4] -> [batch,64,8,8]
            DecoderBlock(128,64),

            # [batch,64,8,8] -> [batch,32,16,16]
            DecoderBlock(64,32),

            # [batch,32,16,16] -> [batch,16,32,32]
            DecoderBlock(32,16),

            # [batch,16,32,32] -> [batch,3,64,64]
            nn.ConvTranspose2d(16,3,kernel_size=2,stride = 2,bias=False),
            # 将输出像素限制在 0~1
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(x)


if __name__ == '__main__':

    # 模拟四张经过预处理的商品图片
    test_images = torch.rand(4,3,IMG_H,IMG_W)

    encoder = ConvEncoder()
    decoder = ConvDecoder()

    # 图片经过编码器得到压缩特征
    encoded_features = encoder(test_images)

    # 压缩特征经过解码器重建图片
    reconstructed_images = decoder(encoded_features)

    print(
        "原始图片形状：",
        test_images.shape,
    )

    print(
        "编码器输出形状：",
        encoded_features.shape,
    )

    print(
        "解码器输出形状：",
        reconstructed_images.shape,
    )

    print(
        "重建结果最小值：",
        reconstructed_images.min().item(),
    )

    print(
        "重建结果最大值：",
        reconstructed_images.max().item(),
    )