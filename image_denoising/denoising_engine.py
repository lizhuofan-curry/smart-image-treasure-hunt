# 本文件负责实现图像去噪模型的一轮训练和验证过程

import torch

def train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
):
    # 完成图像去噪模型的一轮训练
    # 返回 avg_loss

    # 虽然这里面没有用上 Dropout 和 BatchNorm
    # 但还是开启训练模式
    model.train()

    # 累加整个epoch损失
    total_loss = 0.0

    # 记录已经处理的样本数量
    total_samples = 0

    for noisy_images,clean_images in train_loader:

        # 移动到GPU
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # 情况上一次的梯度
        optimizer.zero_grad()

        # 模型根据带噪图片预测噪声
        predicted_noise = model(noisy_images)

        # 计算这批图片的真实噪声
        # 带噪图片 = 清晰图片 + 噪声
        # 所以 真实噪声 = 带噪图片 - 清晰图片
        target_noise = noisy_images-clean_images

        # 比较预测噪声和真实噪声
        loss = criterion(predicted_noise, target_noise)

        # 反向传播
        loss.backward()

        # 限制梯度整体大小，防止某次参数更新突然过大
        # 梯度方向保持不变
        # 整体长度缩短，梯度整体范数 <= 1.0
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # 更新模型参数
        optimizer.step()

        # 当前epoch中实际包含的图片数量
        # 主要是考虑到最后一个batch数量可能不足 batch_size
        batch_size = noisy_images.size(0)

        # MSELoss 默认返回平均损失
        # 计算该epoch的总损失
        total_loss += loss.item() * batch_size

        # 累加样本数量
        total_samples += batch_size
    # 计算整个epoch的平均损失
    return total_loss / total_samples

def evaluate(
        model,
        val_loader,
        criterion,
        device,
):
    # 开启验证模式
    model.eval()

    # 累加验证损失
    total_loss = 0.0

    # 记录验证样本数量‘
    total_samples = 0

    # 验证阶段不需要梯度
    with torch.no_grad():
        for noisy_images,clean_images in val_loader:
            # 移动到GPU
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # 模型预测噪声
            predicted_noise = model(noisy_images)

            # 计算真实噪声
            target_noise = noisy_images-clean_images

            # 比较预测噪声与真实噪声
            loss = criterion(predicted_noise, target_noise)

            # 当前batch 实际数量
            batch_size = noisy_images.size(0)

            # 累加损失和数量
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    # 返回平均损失
    return total_loss / total_samples