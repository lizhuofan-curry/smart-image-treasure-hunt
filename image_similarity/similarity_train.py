# 本文件第一阶段：训练编码器和解码器，保存最佳参数
# 第二阶段：用最佳编码器处理全部 24853 张图片 生成 embedding.npy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import seed_everything

from image_similarity.similarity_config import (
    SEED,                   # 随机种子
    LEARNING_RATE,          # 学习率
    TRAIN_BATCH_SIZE,       # 训练集的批量大小
    VAL_BATCH_SIZE,         # 验证集的批量大小
    FULL_BATCH_SIZE,        # 为全部图片生成特征向量时的批量大小
    EPOCHS,                 # 训练回合
    NUM_WORKERS,            # 多进程数量
    PIN_MEMORY,             # 是否开启锁页内存
    ENCODER_MODEL_PATH,     # 保存编码器参数
    DECODER_MODEL_PATH,     # 保存解码器参数
    EMBEDDING_PATH          # 保存embedding向特征量库
)

from image_similarity.similarity_data import create_dataset
from image_similarity.similarity_model import ConvDecoder,ConvEncoder
from image_similarity.similarity_engine import train_epoch,evaluate

# 使用训练完成的编码器为完整商品图片数据集生成特征向量
# 返回 embedding 形状为 [商品图片数量，特征维度]
def generate_embeddings(
        encoder,
        full_loader,
        device,
):
    # 生成特征时使用验证模式，batchnorm 使用训练期间保存的统计数据
    encoder.eval()

    # 暂时保存每一个 batch 的特征向量
    embedding_batches = []

    # 生成特征时不需要计算梯度
    with torch.no_grad():
        # 这里只是用tqdm包住full_loader,额外显示进度
        for input_images,_ in tqdm(
            full_loader,
            desc = '生成商品特征向量', # 进度条前的说明文字
            unit = 'batch'          # 告诉进度条每前进一步代表处理完一个batch
        ):
            inputs = input_images.to(device)
            # [batch,3,64,64] -> [batch,256,2,2]
            encoded_features = encoder(inputs)

            # [batch,256,2,2] -> [batch,1024]
            features_vectors = torch.flatten(encoded_features,start_dim = 1,)

            # 移动到 cpu 并转换为 numpy 数组
            # numpy 不能直接处理 GPU上的数，因此先移动到 CPU
            features_vectors = features_vectors.cpu().numpy()

            # 接到特征向量后面
            embedding_batches.append(features_vectors)

    # 把所有 batch 沿着样本维度拼接
    # 从第0维开始拼接，因为我们要增加图片数量
    embeddings = np.concatenate(embedding_batches,axis=0)

    # 将完整特征向量库保存到磁盘
    np.save(EMBEDDING_PATH, embeddings)

    return embeddings


# 训练卷积自编码器
# 保存验证损失最低的编码器和解码器
# 最后生成完整商品特征向量库
def train_model():
    # 基本准备
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # 创建数据集
    full_dataset,train_dataset,val_dataset = create_dataset()
    print("完整数据集样本数量：",len(full_dataset),)
    print("训练集样本数量：",len(train_dataset),)
    print("验证集样本数量：",len(val_dataset),)

    # 创建 dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and torch.cuda.is_available()),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and torch.cuda.is_available()),
    )

    # 完整数据集用来生成 embeddings.npy
    full_loader = DataLoader(
        full_dataset,
        batch_size=FULL_BATCH_SIZE,
        # 必须保持自然排序后的原始顺序
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and torch.cuda.is_available()),
    )

    # 创建模型
    encoder = ConvEncoder().to(device)
    decoder = ConvDecoder().to(device)

    # 使用均方误差比较重建图片和原始图片
    criterion = nn.MSELoss()

    # 将解码器和编码器一起交给优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LEARNING_RATE,
    )

    # 训练记录

    train_losses= []
    val_losses = []

    # 当前最小验证损失
    best_val_loss = float('inf')

    # 最佳模型所在的epoch
    best_epoch = 0

    epoch_progress = tqdm(
        range(EPOCHS),
        total=EPOCHS,
        desc = '相似图片自编码器训练',
        unit = 'epoch',
    )
    for epoch in epoch_progress:
        # 训练一个epoch

        train_loss = train_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # 验证一个 epoch
        val_loss = evaluate(
            encoder,
            decoder,
            val_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 在进度条右侧显示当前指标
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
        )

        tqdm.write(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # 验证损失下降时保存两个模型
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            # 保存最佳编码器参数
            torch.save(
                # 取得编码器当前所有需要保存的参数和状态，类似于字典结构
                encoder.state_dict(),
                ENCODER_MODEL_PATH,
            )

            # 保存最佳解码器参数
            torch.save(
                decoder.state_dict(),
                DECODER_MODEL_PATH,
            )
            tqdm.write(
                "验证损失降低，编码器和解码器已保存。"
            )

    # 加载最佳编码器
    print('自编码器训练完成')
    print('最佳模型所在轮次：', best_epoch)
    print('最佳验证损失：',f'{best_val_loss:.6f}')

    # 当前内存中 encoder 是最后一轮参数，不一定是验证最低的参数
    # 加载最佳参数，运用到生成特征向量库

    # 这一步只是从磁盘中读取字典，还没装进模型
    best_encoder_state = torch.load(ENCODER_MODEL_PATH, map_location=device)

    # 这一步表示把文件中保存的参数，逐层安装回当前创建好的编码器
    encoder.load_state_dict(best_encoder_state)

    print('已重新加载最佳编码器')

    # 生成 embedding 特征库
    embeddings = generate_embeddings(
        encoder,
        full_loader,
        device,
    )
    print(
        "特征向量库形状：",
        embeddings.shape,
    )

    print(
        "特征向量库已保存到：",
        EMBEDDING_PATH,
    )

    print(
        "第 0 行特征对应图片：",
        full_dataset.image_paths[0].name,
    )

    return (
        train_losses,
        val_losses,
        embeddings,
    )

if __name__ == "__main__":
    train_model()
