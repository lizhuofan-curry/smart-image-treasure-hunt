# 本文件负责定义带跳跃连接的 U - Net 图像去噪模型

import torch
import torch.nn as nn

'''
    U-Net 中使用的基础卷积块
    输入：
        [batch,in_channels,H,W]
    输出：
        [batch,out_channels,H,W]
    两次卷积都使用：
        kernel_size = 3
        stride = 1
        padding = 1
    因此不会改变图片的高度和宽度
    只会改变通道数量并进一步提取特征
    '''
class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # 第一次提取特征并改变通道数
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                # 后面有 BatchNorm ,所以卷积偏置可以关闭
                bias = False,
            ),
            # 控制当前通道特征的数值分布
            # 防止特征值随着网络加深不断放大
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次继续提取当下尺度的特征
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

'''
SE 通道注意力模块

输入和输出形状保持不变
[batch,channels,heights,width]
'''
class SEBlock(nn.Module):
    def __init__(self,
                 channels,
                 reduction = 16
    ):
        super().__init__()

        hidden_channels = max(channels // reduction, 4)

        # Squeeze : 把每个通道压缩成一个数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation 计算每个通道的重要程度
        self.channel_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size = 1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size = 1,
            ),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # 得到每个通道的权重
        channel_weight = self.avg_pool(x)
        channel_weight = self.channel_attention(channel_weight)

        # 对每个通道进行加权
        return x * channel_weight


'''
    简化版 U-Net 残差图像去噪模型
    输入：
        带噪 RGB 图片
        [batch,3,64,64]
    输出：
        模型预测的噪声残差
        [batch,3,64,64]
    核心结构：
        编码器负责提取特征并逐步缩小尺寸
        解码器负责逐步恢复图片尺寸
        跳跃连接把编码器中的空间细节直接传给解码器
    '''

class ConvDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

        # 1.编码器
        # 第一层不缩小图片，只提取浅层特征
        # 浅层特征保存较多：边缘，颜色，人物位置和局部纹理
        # [batch,3,64,64] -> [batch,32,64,64]

        self.encoder1 = ConvBlock(3,32)
        # 将宽高缩小一半 64x64 -> 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        # [batch,32,32,32] -> [batch,64,32,32]
        self.encoder2 = ConvBlock(32,64)
        # [batch,64,32,32] -> [batch,64,16,16]
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        # [batch,64,16,16] -> [batch,128,16,16]
        self.encoder3 = ConvBlock(64,128)
        # [batch,128,16,16] -> [batch,128,8,8]
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        # 2.瓶颈层：位于 U-Net 最底部
        # 负责学习图片的整体语义和高级特征
        # [batch,128,8,8] -> [batch,256,8,8]
        self.bottleneck = ConvBlock(128,256)

        # 对最深层的 256 个特征通道重新分配权重
        self.bottleneck_se = SEBlock(256,16)

        # 解码器第一阶段
        # 先将尺寸从 8x8 放大到 16x16
        # 再通过 1x1 卷积把通道数从 256 降到 128
        # [batch,256,8,8] -> [batch,128,16,16]
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256,128,1),
        )

        # up3的输出有 128 个通道，encoder3 的跳跃特征也有128个通道
        # 拼接之后： 128+128 = 256个通道
        self.decoder3 =ConvBlock(256,128)


        # 解码器第二阶段
        # [batch,128,16,16] -> [batch,64,32,32]
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128,64,1)
        )

        # 拼接后 [batch,128,32,32] -> [batch,64,32,32]
        self.decoder2 = ConvBlock(128,64)

        # 解码器第三阶段
        # [batch,64,32,32] -> [batch,32,64,64]
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64,32,1)
        )

        # 拼接后 [batch,64,64,64] -> [32,64,64]
        self.decoder1 = ConvBlock(64,32)

        # 输出层
        # 将 32 个特征通道转换成 3 通道噪声残差
        # 注意：噪声既可能是正数也可能是负数
        # 因此这里不能用 sigmoid
        self.output_layer = nn.Conv2d(32,3,1)

    def forward(self, x):
        # 编码阶段

        # 保存 64x64的浅层特征 以后传给最后一个解码阶段
        # [batch,3,64,64] -> [batch,32,64,64]
        enc1 = self.encoder1(x)
        # [batch,32,64,64] -> [batch,32,32,32]
        x = self.pool1(enc1)

        # [batch,32,32,32] -> [batch,64,32,32]
        enc2 = self.encoder2(x)
        # [batch,64,32,32] -> [batch,64,16,16]
        x = self.pool2(enc2)

        # [batch,64,16,16] -> [batch,128,16,16]
        enc3 = self.encoder3(x)
        # [batch,128,16,16] -> [batch,128,8,8]
        x = self.pool3(enc3)

        # 瓶颈阶段
        # [batch,128,8,8] -> [batch,256,8,8]
        x = self.bottleneck(x)

        # 对瓶颈层的 256 个通道重新分配权重
        # 输入输出都是 [batch,256,8,8]
        bottleneck = self.bottleneck_se(x)

        # 解码阶段一
        # [batch,256,8,8] -> [batch,128,16,16]
        x = self.up3(bottleneck)

        # 将解码器特征与 encoder3 的特征沿通道维度拼接
        # [batch,128,16,16] -> [batch,256,16,16]
        x = torch.cat([x,enc3],dim=1)
        # [batch,256,16,16] -> [batch,128,16,16]
        x = self.decoder3(x)

        # 解码阶段二
        # [batch,128,16,16] -> [batch,64,32,32]
        x = self.up2(x)

        # 与 encoder2 沿通道维度拼接
        # [batch,64,32,32] -> [batch,128,32,32]
        x = torch.cat([x,enc2],dim=1)
        # [batch,128,32,32] -> [batch,64,32,32]
        x = self.decoder2(x)

        # 解码阶段三
        # [batch,64,32,32] -> [batch,32,64,64]
        x = self.up1(x)

        # 与encoder1 拼接
        # [batch,32,64,64] -> [batch,64,64,64]
        x = torch.cat([x,enc1],dim=1)
        # [batch,64,64,64] -> [batch,32,64,64]
        x = self.decoder1(x)

        # 模型不再输出清晰图片，而是预测带噪图片中的声音残差
        predicted_noise = self.output_layer(x)
        return predicted_noise


# 当前阶段只测试模型前向传播和形状变化
if __name__ == "__main__":

    # 模拟一个包含 4 张图片的 batch
    #
    # 形状：
    # [batch_size, channels, height, width]
    test_input = torch.rand(
        4,
        3,
        64,
        64,
    )

    # 创建去噪模型
    model = ConvDenoiser()

    # 执行前向传播
    test_output = model(test_input)


    print("模型输入形状：", test_input.shape)
    print("模型输出形状：", test_output.shape)

    print(
        "模型输出像素范围：",
        test_output.min().item(),
        test_output.max().item(),
    )