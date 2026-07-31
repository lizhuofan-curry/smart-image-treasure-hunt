# 本文件负责加载商品分类模型，评估验证集表现并可视化分类结果

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.utils import seed_everything

from image_classification.classification_config import (
    SEED,
    TEST_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    CLASSIFIER_MODEL_PATH,
    CLASSIFICATION_NAMES,
)

from image_classification.classification_data import (
    create_dataset,
)

from image_classification.classification_model import (
    Classifier,
)

from image_classification.classification_engine import (
    evaluate,
)


# 让 matplotlib 可以正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]

# 避免坐标轴中的负号显示成方块
plt.rcParams["axes.unicode_minus"] = False


def show_predictions(
        model,
        data_loader,
        device,
):
    """
    从验证集中取出一个 batch，
    展示图片的真实类别和预测类别。
    """

    # 切换到验证模式
    # BatchNorm 使用训练阶段保存的统计数据
    model.eval()

    # 从 DataLoader 中取得第一个 batch
    images, labels = next(
        iter(data_loader)
    )

    # 将图片移动到模型所在设备
    images = images.to(device)

    # 推理阶段不需要计算梯度
    with torch.no_grad():

        # outputs 形状为 [batch, 5]
        outputs = model(images)

        # 在每张图片的 5 个类别分数中，
        # 找到分数最大值所在的位置
        predictions = torch.argmax(
            outputs,
            dim=1,
        )

    print(
        "模型输出形状：",
        outputs.shape,
    )

    print(
        "预测标签形状：",
        predictions.shape,
    )

    # matplotlib 只能直接处理 CPU 数据
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    # 最多显示 8 张图片
    display_count = min(
        8,
        images.size(0),
    )

    # 创建 2 行 4 列的画布
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 7),
    )

    # 将二维 axes 展平成一维，
    # 方便使用 axes[i] 逐个访问
    axes = axes.flatten()

    for index in range(display_count):

        # Tensor 图片原形状为 [C, H, W]
        # imshow 需要 [H, W, C]
        image = images[index].permute(
            1,
            2,
            0,
        ).numpy()

        # 获取真实类别编号
        true_label = labels[index].item()

        # 获取预测类别编号
        predicted_label = (
            predictions[index].item()
        )

        # 根据类别编号取得中文类别名称
        true_name = (
            CLASSIFICATION_NAMES[
                true_label
            ]
        )

        predicted_name = (
            CLASSIFICATION_NAMES[
                predicted_label
            ]
        )

        # 判断本张图片是否预测正确
        is_correct = (
            true_label
            == predicted_label
        )

        # 预测正确使用绿色标题，
        # 预测错误使用红色标题
        title_color = (
            "green"
            if is_correct
            else "red"
        )

        axes[index].imshow(image)

        axes[index].set_title(
            f"真实：{true_label}-{true_name}\n"
            f"预测：{predicted_label}-{predicted_name}",
            color=title_color,
            fontsize=11,
        )

        axes[index].axis("off")

    # 当实际显示图片少于 8 张时，
    # 隐藏没有使用的子图
    for index in range(
            display_count,
            len(axes),
    ):
        axes[index].axis("off")

    plt.tight_layout()
    plt.show()


def test_model():
    """
    加载验证准确率最高的模型参数，
    计算验证集指标并展示预测结果。
    """

    # ==================== 1. 基本准备 ====================

    # 固定随机种子，
    # 保证数据集划分与训练时一致
    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("当前设备：", device)


    # ==================== 2. 创建验证集 ====================

    _, val_dataset = create_dataset()

    print(
        "验证集样本数量：",
        len(val_dataset),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )


    # ==================== 3. 加载最佳模型 ====================

    if not CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(
            "没有找到分类模型参数："
            f"{CLASSIFIER_MODEL_PATH}，"
            "请先运行 classification_train.py"
        )

    # 创建与训练阶段完全相同的模型结构
    model = Classifier().to(device)

    # 从本地读取最佳参数字典
    model_state = torch.load(
        CLASSIFIER_MODEL_PATH,
        map_location=device,
    )

    # 将参数加载进模型
    model.load_state_dict(model_state)

    print(
        "分类模型加载完成：",
        CLASSIFIER_MODEL_PATH,
    )


    # ==================== 4. 计算验证集指标 ====================

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"验证集平均损失：{val_loss:.6f}"
    )

    print(
        f"验证集分类准确率：{val_acc:.4f}"
    )

    print(
        f"验证集分类准确率：{val_acc * 100:.2f}%"
    )


    # ==================== 5. 显示预测结果 ====================

    show_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
    )


if __name__ == "__main__":
    test_model()