# 本文件负责实现商品分类模型的一轮训练过程和验证过程

import torch

# 训练一个 epoch ,并返回平均训练损失
def train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
):
    # 开启训练模式
    # BatchNorm 使用当前 batch 的统计数据
    model.train()

    # 记录当前 epoch 的累加损失
    total_loss = 0.0

    # 记录当前 epoch 处理的样本总数
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 向前传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        # 当前 batch 中图片数量
        batch_size = images.size(0)

        # loss.item () 是当前 batch 的平均损失，乘上 batch_size 得到当前batch的总损失
        total_loss += loss.item() * batch_size

        total_samples += batch_size

    # 整个 epoch 所有样本的平均损失
    return total_loss / total_samples

# 验证模型，并返回平均验证损失和分类准确率
def evaluate(
        model,
        val_loader,
        criterion,
        device,
):
    # 开启验证模式
    # batchnorm 使用训练阶段保存的统计数据
    model.eval()

    # 积累总损失
    total_loss = 0.0
    # 处理总样本数
    total_samples = 0

    # 记录预测正确的图片数量
    correct_samples = 0

    # 验证阶段不需要梯度
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 向前传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 当前 batch 的数量
            batch_size = images.size(0)

            # 当前 batch 的总损失
            total_loss += loss.item() * batch_size

            total_samples += batch_size

            # output 为 [batch,5],在五个类别分数中找到最大的位置
            predictions = torch.argmax(outputs, dim=1)

            # 比较预测类别和真实标签
            correct_samples += (predictions == labels).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = correct_samples / total_samples
    return avg_loss, avg_acc
