# 本文件负责定义商品分类模型，并将商品图片预测为 5 个类别之一

import torch
import torch.nn as nn

from image_classification.classification_config import IMG_H,IMG_W,NUM_CLASSES

class CovnBlock(nn.Module):
    # 一个卷积特征提取块
    # 输入：[batch,in_channels,H,W]
    # 输出：[batch,out_channels,H/2,W/2]

    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()

        # 提取图片的局部特征
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 稳定不同batch中的特征分布，标准化
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            # 高度和宽度缩小一半
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)

# 商品分类模型
class Classifier(nn.Module):
    def __init__(
            self,
            num_classes=NUM_CLASSES,
    ):
        super().__init__()

        # 增加通道数量，同时通过最大池化来逐渐缩小尺寸

        self.feature_extractor = nn.Sequential(
            # [batch,3,64,64] -> [batch,16,32,32]
            CovnBlock(3,16),

            # [batch,16,32,32] -> [batch,32,16,16]
            CovnBlock(16,32),

            # [batch,32,16,16] -> [batch,64,8,8]
            CovnBlock(32,64),

            # [batch,64,8,8] -> [batch,128,4,4]
            CovnBlock(64,128),
        )

        # 把每个通道的 4x4 特征图压缩为 1x1
        # [batch,128,4,4] -> [batch,128,1,1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 根据提取出的 128 个特征，输出每个商品的预测分数
        self.classifier = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, x):
        # 卷积网络提取图片特征，提炼出更重要的信息
        # [batch,3,64,64] -> [batch,128,4,4]
        x = self.feature_extractor(x)

        # 全局平均池化，将128个特征图变成128个特征值
        # [batch,128,4,4] -> [batch,128,1,1]
        x = self.global_pool(x)

        # 改变形状，将四维变成二维，方便全连接层变换，1指从第一维开始展平
        # [batch,128,1,1] -> [batch,128]
        x = torch.flatten(x, 1)

        # [batch,128] -> [batch,5]
        outputs = self.classifier(x)

        return outputs


if __name__ == "__main__":

    # 模拟一个包含 4 张商品图片的 batch
    test_images = torch.rand(
        4,
        3,
        IMG_H,
        IMG_W,
    )

    model = Classifier()

    test_outputs = model(test_images)

    print(
        "模型输入形状：",
        test_images.shape,
    )

    print(
        "模型输出形状：",
        test_outputs.shape,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "模型参数数量：",
        total_parameters,
    )

