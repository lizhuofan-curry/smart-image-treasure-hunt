# 本文件负责实现自编码器的一轮训练过程和验证过程

import torch
from sklearn.neighbors import NearestNeighbors

def train_epoch(
        encoder,
        decoder,
        train_loader,
        criterion,
        optimizer,
        device,
):
    # 训练编码器和解码器的一个epoch
    # 返回当前 epoch 的平均损失

    # 开启训练模式
    encoder.train()
    decoder.train()

    # 累加所有样本的损失
    total_loss = 0.0

    # 记录处理样本总数
    total_samples = 0

    for input_images,target_images in train_loader:
        # 移动到 CPU 或者 GPU
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        # 清空上一个batch的梯度
        optimizer.zero_grad()

        # 编码器压缩图片
        # [batch,3,64,64] -> [batch,256,2,2]
        encoded_features = encoder(input_images)

        # 解码器根据压缩特征重建图片
        # [batch,256,2,2] -> [batch,3,64,64]
        reconstructed_images = decoder(encoded_features)

        # 比较重建图片和原始目标图片
        loss = criterion(reconstructed_images,target_images)

        # 反向传播
        loss.backward()

        # 更新编码器和解码器参数
        # 在优化器中把编码器和解码器的参数都写进去了
        optimizer.step()

        # 当前batch 的图片数量
        batch_size = input_images.size(0)

        # loss.item()是当前batch的平均损失
        # 乘上当前batch的数量得到当前batch的总损失
        total_loss += loss.item() * batch_size

        total_samples += batch_size
    # 返回这个 epoch 的平均训练损失
    return total_loss / total_samples

def evaluate(
        encoder,
        decoder,
        val_loader,
        criterion,
        device,
):
    # 在验证集上评估编码器和解码器
    # 返回平均验证损失

    # 开启验证模式
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_samples = 0

    # 验证阶段不计算梯度
    with torch.no_grad():
        for input_images,target_images in val_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            # 提取压缩特征
            encoded_features = encoder(input_images)

            # 重建图片
            reconstructed_images = decoder(encoded_features)

            # 计算重建损失
            loss = criterion(reconstructed_images,target_images)

            batch_size = input_images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


#  增加 KNN 相似图片检索函数

'''
使用编码器提取查询图片特征，并从完整商品特征库中寻找最相似的图片
参数：
    encoder:已经训练好的编码器
    image_tensor:查询图片tensor 形状为[batch,3,64,64]
    num_images: 需要返回的相似图片数量
    embeddings: 完整商品图片特征库 [24853,1024]
返回：
    distances: 查询图片与近邻图片的余弦距离
    indices:相似图片在完整数据集中的索引
'''
def compute_similar_images(
        encoder,
        image_tensor,
        num_images,
        embeddings,
        device,
):
    # 查询图片必须带有 batch 维度
    # 虽然getitem返回的是三维
    # 但是 Dataloader 会自动增加一个维度，所以不需要我们手动增加
    # 当单独上传一张图片时则需要手动增加batch维度 ,unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError(
            '查询图片形状必须为[batch,3,64,64]'
            f'当前形状不符合'
        )
    # 特征库必须是二维矩阵
    if embeddings.ndim != 2:
        raise ValueError(
            'embedding 必须是二维矩阵'
            f'当前形状为{embeddings.shape}'
        )

    if num_images <= 0:
        raise ValueError(
            "num_images 必须大于 0"
        )

    if num_images > len(embeddings):
        raise ValueError(
            "需要返回的图片数量不能超过特征库中的图片数量"
        )

    # 使用验证模式
    # BatchNorm 使用训练阶段保存的统计数据
    encoder.eval()

    image_tensor = image_tensor.to(device)

    # 查询阶段不计算梯度
    with torch.no_grad():
        # [batch,3,64,64] -> [batch,256,2,2]
        encoded_features = encoder(image_tensor)

        # [batch,256,2,2] -> [batch,1024]
        image_vectors = torch.flatten(encoded_features,1)

        # KNN 使用 numpy 数组
        image_vectors = image_vectors.cpu().numpy()

    # 查询图片和特征库中的特征维度必须一致
    if image_vectors.shape[1] != embeddings.shape[1]:
        raise RuntimeError(
            "查询图片特征维度与商品特征库不一致："
            f"查询特征维度为 {image_vectors.shape[1]}，"
            f"特征库维度为 {embeddings.shape[1]}"
        )

    # 创建最近邻检索器
    knn = NearestNeighbors(
        n_neighbors=num_images, # 要返回的数量
        metric='cosine',        # cosine 表示使用余弦距离比较特征向量
    )

    # 完整商品特征库登记到KNN中
    knn.fit(embeddings)

    # 在商品特征库中搜素最近的 num_images 个向量
    distances,indices = knn.kneighbors(image_vectors)

    return distances,indices
