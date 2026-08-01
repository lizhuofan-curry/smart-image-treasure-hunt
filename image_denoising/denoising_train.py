# 本文件负责组织图像去噪模型的数据加载、训练、验证和模型保存流程

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from common.utils import seed_everything
from image_denoising.denoising_config import (
    SEED,                   # 随机种子
    LEARNING_RATE,          # 学习率
    TRAIN_BATCH_SIZE,       # 训练集大小
    VAL_BATCH_SIZE,         # 验证集大小
    EPOCHS,                 # 训练轮数
    DENOISER_MODEL_PATH     # 最佳模型保存路径
)

from image_denoising.denoising_data import create_dataset
from image_denoising.denoising_engine import train_epoch,evaluate
from image_denoising.denoising_model import ConvDenoiser

# 完成图像去噪模型的完整训练流程
def train_model():

    # 1。固定随机种子：让数据集划分，模型初始化等尽量可复现
    seed_everything(SEED)

    # 2.设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('当前训练设备：', device)

    # 3.创建训练集和验证集
    train_dataset,val_dataset = create_dataset()

    print('训练集图片数量：',len(train_dataset))
    print('验证集图片数量：',len(val_dataset))

    # 4.创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        # windows 下优先单进程读取，避免多进程启动问题
        num_workers=0,
        # 最后一个 batch 数量不足时仍保留
        drop_last=False,

    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        # 验证集不训练，因此不需要打乱顺序
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    print("训练集 batch 数量：", len(train_loader))
    print("验证集 batch 数量：", len(val_loader))

    # 5. 创造模型
    model = ConvDenoiser()
    # 模型与图片必须位于相同设备
    model=model.to(device)

    # 6.创建损失函数
    # MSELoss 计算的是误差平方，为了降低整体误差，模型有时会倾向于把多个可能的像素取平均
    # 结果就是噪声点少了，但是边缘，图案和纹理也模糊了
    #  L1Loss 使用绝对误差 |预测像素 - 真实像素|
    # 更有利于保留 商品轮廓，衣服边缘，花纹等
    criterion = nn.L1Loss()

    # 7. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8.开始训练

    # 记录目前出现的最低验证损失
    best_val_loss =float('inf')

    # 保存每一个 Epoch 的损失，后面绘制训练曲线时使用
    train_losses = []
    val_losses = []

    # 创建 Epoch 进度条
    epoch_progress = tqdm(
        range(EPOCHS),
        total=EPOCHS,
        desc="图像去噪模型训练",
        unit="epoch",
    )

    for epoch in epoch_progress:

        # 完成一轮训练
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        # 在验证集上评估当前模型
        val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 在进度条右侧显示当前损失
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            best_val_loss=f"{best_val_loss:.6f}",
        )

        # 输出当前 Epoch 的完整训练记录
        tqdm.write(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )


        # 9. 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # state_dict() 保存的是模型中所有可训练参数
            torch.save(
                model.state_dict(),
                DENOISER_MODEL_PATH,
            )

            tqdm.write(
                "保存新的最佳模型，"
                f"验证损失：{best_val_loss:.6f}"
            )

    print("训练完成。")
    print("最佳验证损失：", best_val_loss)
    print("模型保存路径：", DENOISER_MODEL_PATH)

    return train_losses, val_losses

# 只有直接运行本文件时才启动训练
if __name__ == "__main__":
    train_model()
