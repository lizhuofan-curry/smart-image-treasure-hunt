# 本文件负责加载编码器和商品特征库，并测试相似图片检索效果

from pathlib import Path  # 把普通路径字符转换成路径对象
# Tk负责创建一个桌面窗口小程序
# filedialog 负责弹出选择文件的窗口  在html中用 <input type = 'file'>把图片上传flask
from tkinter import filedialog,Tk   # tkinter 是Python 自带的桌面窗口工具
from PIL import Image               # 用于打开用户图片

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.utils import seed_everything

from image_similarity.similarity_config import (
    SEED,
    ENCODER_MODEL_PATH,
    EMBEDDING_PATH,
    NUM_SIMILAR_IMAGES,
)

from image_similarity.similarity_data import (
    create_dataset,create_transform
)

from image_similarity.similarity_model import (
    ConvEncoder,
)

from image_similarity.similarity_engine import (
    compute_similar_images,
)


# 让 matplotlib 尽量正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False



'''
弹出文件选择窗口，读取一张外部图片
并进行与训练阶段完成相同的预处理
返回：
    image_path 所选图片的路径
    image_tensor [3,64,64] 图片的Tensor
'''
def select_query_image():
    # 创建 tkinter 窗口，但不显示主窗口
    root = Tk()     # 创建窗口系统
    root.withdraw() # 隐藏 Tkinter 主窗口

    # 让文件选择窗口显示在其他窗口前面
    root.attributes("-topmost", True)

    # askopenfilename() 弹出文件选择窗口 -> 等待用户选择文件 -> 返回所选文件的完整路径
    file_path = filedialog.askopenfilename(
        title = '请选择需要检索的商品图片',     # 设置窗口顶部显示的标题
        filetypes = [
            (
                "图片文件",
                "*.jpg *.jpeg *.png *.bmp",
            ),
            (
                "所有文件",
                "*.*",
            ),
            ],
    )
    root.destroy()  # 用户选择完文件后，关闭刚刚创建的 Tkinter 窗口系统

    # 用户关闭窗口或者没有选择图片
    if not file_path:
        raise RuntimeError('没有选择查询图片')

    image_path = Path(file_path) # 字符串转为 Path
    if not image_path.exists():
        raise FileNotFoundError(f'没有找到查询照片{image_path}')

    # 使用与训练阶段相同的预处理
    transform = create_transform()
    with Image.open(image_path) as image:  # 使用 with 后，代码块结束时会自动关闭窗口图片文件
        image = image.convert('RGB')
        image_tensor = transform(image)
    return image_path,image_tensor


def test_similarity_search():
    # ==================== 1. 基本准备 ====================

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "当前设备：",
        device,
    )


    # ==================== 2. 创建完整数据集 ====================

    full_dataset, _, _ = create_dataset()

    print(
        "完整数据集图片数量：",
        len(full_dataset),
    )


    # ==================== 3. 选择查询图片 ====================

    # 从电脑中选择一张外部查询图片
    query_path,query_image = select_query_image()
    print('查询图片路径：',query_path)

    print(
        "增加 batch 前的形状：",
        query_image.shape,
    )

    # 增加 batch 维度
    # [3, 64, 64] → [1, 3, 64, 64]
    query_batch = query_image.unsqueeze(0)

    print(
        "增加 batch 后的形状：",
        query_batch.shape,
    )


    # ==================== 4. 加载最佳编码器 ====================

    if not ENCODER_MODEL_PATH.exists():
        raise FileNotFoundError(
            "没有找到编码器参数："
            f"{ENCODER_MODEL_PATH}，"
            "请先运行 similarity_train.py"
        )

    encoder = ConvEncoder().to(device)

    encoder_state = torch.load(
        ENCODER_MODEL_PATH,
        map_location=device,
    )

    encoder.load_state_dict(
        encoder_state
    )

    encoder.eval()

    print(
        "编码器加载完成：",
        ENCODER_MODEL_PATH,
    )


    # ==================== 5. 加载商品特征库 ====================

    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            "没有找到商品特征向量库："
            f"{EMBEDDING_PATH}，"
            "请先运行 similarity_train.py"
        )

    embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        "商品特征向量库形状：",
        embeddings.shape,
    )

    # 特征库行数应当和完整数据集图片数量相同
    if len(embeddings) != len(full_dataset):
        raise RuntimeError(
            "特征向量数量与商品图片数量不一致："
            f"特征向量数量为 {len(embeddings)}，"
            f"商品图片数量为 {len(full_dataset)}"
        )


    # ==================== 6. 检索相似图片 ====================

    # 查询图片本身就在完整特征库中，
    # 因此多查找一张，后面把自己排除
    search_count = NUM_SIMILAR_IMAGES

    distances, indices = (
        compute_similar_images(
            encoder=encoder,
            image_tensor=query_batch,
            num_images=search_count,
            embeddings=embeddings,
            device=device,
        )
    )

    # 当前只查询一张图片
    # 所以取第 0 行结果
    distances = distances[0]
    indices = indices[0]

    # 排除查询图片自己
    similar_results = []

    for index, distance in zip(
            indices,
            distances,
    ):
        index = int(index)
        distance = float(distance)

        similar_results.append(
            (index, distance)
        )

        if (
            len(similar_results)
            == NUM_SIMILAR_IMAGES
        ):
            break


    print("相似图片检索结果：")

    for rank, (
            image_index,
            distance,
    ) in enumerate(
            similar_results,
            start=1,
    ):
        image_name = (
            full_dataset
            .image_paths[image_index]
            .name
        )

        print(
            f"第 {rank} 名："
            f"索引 {image_index}，"
            f"文件 {image_name}，"
            f"余弦距离 {distance:.6f}"
        )


    # ==================== 7. 显示检索结果 ====================

    # 一张查询图片，加上五张相似图片
    figure_count = (
        NUM_SIMILAR_IMAGES
        + 1
    )

    fig, axes = plt.subplots(
        nrows=2,          # 两行三列
        ncols=3,
        figsize=(24, 6),  # 24 宽度 6 高度
        dpi = 120         # dpi 显示清晰度
    )
    axes = axes.flatten()
    # 显示查询图片
    query_display = (
        query_image
        .permute(1, 2, 0)
        .numpy()
    )

    axes[0].imshow(
        query_display
    )

    axes[0].set_title(
        "外部查询图片\n"
        f"{query_path.name}",
        fontsize=14,
    )

    axes[0].axis("off")

    # 显示检索到的相似图片
    for position, (
            image_index,
            distance,
    ) in enumerate(
            similar_results,
            start=1,
    ):
        similar_image, _ = (
            full_dataset[
                image_index
            ]
        )

        similar_display = (
            similar_image
            .permute(1, 2, 0)
            .numpy()
        )

        image_name = (
            full_dataset
            .image_paths[image_index]
            .name
        )

        axes[position].imshow(
            similar_display
        )

        axes[position].set_title(
            f"相似图片 {position}\n"
            f"{image_name}\n"
            f"距离：{distance:.4f}",
            fontsize=14,
        )

        axes[position].axis("off")

    plt.tight_layout()

    # 这里只显示，不保存静态结果图片
    plt.show()


if __name__ == "__main__":
    test_similarity_search()