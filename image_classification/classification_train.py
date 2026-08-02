# 本文件负责完成商品分类模型从数据加载，训练，验证到参数保存的完整流程

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.utils import seed_everything

from image_classification.classification_config import (
    SEED,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    LEARNING_RATE,
    EPOCHS,
    CLASSIFIER_MODEL_PATH,
    NUM_CLASSES,
    CLASSIFICATION_NAMES,
)

from image_classification.classification_model import Classifier
from image_classification.classification_data import create_dataset
from image_classification.classification_engine import train_epoch,evaluate

# 这里传入的是训练集，最终返回的是每类训练图片的数量已经每类对应的损失权重
def calculate_class_weights(train_dataset):
    """
    根据当前训练集中的类别数量计算交叉熵权重。

    公式：总样本数 / (类别数 * 当前类别样本数)
    少数类别得到更大的权重，多数类别得到更小的权重。
    """
    train_labels = []
    for index in train_dataset.indices:
        # train_dataset.dataset 取得原始的完整数据集对象  这是 Subset 的属性
        # 根据图片编号找到对应的标签
        label = train_dataset.dataset.labels_dict[index]
        train_labels.append(label)

    train_labels = torch.tensor(train_labels)

    # 统计每类图片数量
    # torch.bincount 用来统计每个数字出现了多少次
    # minlength 表示结构必须至少包含五个位置
    class_counts = torch.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    )

    # 检查是否缺少类别
    if torch.any(class_counts == 0):
        missing_classes = torch.where(
            class_counts == 0
        )[0].tolist()
        raise ValueError(
            "训练集中缺少类别："
            f"{missing_classes}，无法计算类别权重"
        )
    # 计算类别权重
    # 类别权重等于 训练集总图片数 / 类别总数 x 当前类别图片数
    # 意思是图片越多，权重越小，图片越少，权重越大，主要是用来解决上衣和下衣的问题
    # 该数据集中上衣有 9000左右张，而下衣只有 1500 左右
    # 使模型更加重视数量较少的类别
    class_weights = (
        len(train_labels)
        / (
            NUM_CLASSES
            * class_counts.float()
        )
    )

    return class_counts, class_weights


def train_model():
    # 完成商品分类模型的训练和验证
    # 并保存验证集准确率最高的模型参数
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('当前设备：', device)

    # 创建数据集
    train_dataset,val_dataset  = create_dataset()

    # 创建 dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        # 训练集需重新打乱顺序
        shuffle=True,
        num_workers=NUM_WORKERS,
        # 使用 CUADA 时开启锁页内存，可以加快数据从 CPU传输到 GPU
        pin_memory=(PIN_MEMORY and device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        # 验证集不用打乱顺序，便于复现
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and device.type == 'cuda'),
    )

    # 创建模型
    model = Classifier().to(device)

    # 根据训练集类别数量自动计算权重。
    # 权重只用于训练，减少多数类别“上衣”对梯度的主导作用。
    class_counts, class_weights = calculate_class_weights(
        train_dataset
    )

    print("训练集类别数量与损失权重：")
    for class_index in range(NUM_CLASSES):
        print(
            f"  {CLASSIFICATION_NAMES[class_index]}："
            f"{class_counts[class_index].item()} 张，"
            f"权重 {class_weights[class_index].item():.4f}"
        )

    # 创建加权交叉熵
    # 上衣数据很多，可以降低单张上衣的影响；下装、包和手表数据较少，需要提高它们预测错误时的训练惩罚
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    # 验证集继续使用普通交叉熵，保证验证损失可与旧实验比较。
    val_criterion = nn.CrossEntropyLoss()

    # Adam 根据参数梯度自动更新模型
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练记录

    # 记录每个 epoch 的训练损失
    train_losses = []

    # 记录每个 epoch 的验证损失
    val_losses = []

    # 记录每个 epoch 的验证准确率
    val_accuracies = []

    # 当前最高验证准确率
    best_val_acc = 0.0

    # 最高准确率对应的验证损失
    best_val_loss = float("inf")

    # 开始训练
    # 在控制台显示训练记录，不会生成记录文件
    epoch_progress = tqdm(
        range(EPOCHS),
        total=EPOCHS,
        desc='商品分类模型训练',
        unit = 'epoch',
    )

    for epoch in epoch_progress:
        # 训练一个完整的epoch
        train_loss = train_epoch(
            model = model,
            train_loader = train_loader,
            criterion = train_criterion,
            optimizer = optimizer,
            device = device,
        )

        # 在验证集上计算 损失和准确率
        val_loss,val_acc = evaluate(
            model = model,
            val_loader = val_loader,
            criterion = val_criterion,
            device = device
        )

        # 在进度条右侧显示当前训练指标
        '''
        商品分类模型训练: 90%|█████████ | 9/10
        [train_loss=0.016167, val_loss=0.031982, val_acc=0.9902]
        '''
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            val_acc=f"{val_acc:.4f}",
        )

        # 保存当前 epoch 的训练记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        tqdm.write(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # 验证准确率提高时，保存当前模型参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CLASSIFIER_MODEL_PATH,
            )
            tqdm.write(
                "验证准确率提高，"
                f"模型已保存到：{CLASSIFIER_MODEL_PATH}"
            )

    # 训练完成
    print("商品分类模型训练完成")
    print(
        "最佳验证准确率：",
        f"{best_val_acc:.4f}",
    )

    print(
        "最佳模型对应验证损失：",
        f"{best_val_loss:.6f}",
    )
    return train_losses,val_losses,val_accuracies

if __name__ == "__main__":
    train_model()
