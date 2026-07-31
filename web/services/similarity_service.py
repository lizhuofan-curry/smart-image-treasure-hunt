# 本文件负责加载相似检索编码器和商品特征库，并检索相似商品图片

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors

from image_similarity.similarity_config import (
    EMBEDDING_PATH,
    ENCODER_MODEL_PATH,
    IMG_DIR,
    NUM_SIMILAR_IMAGES,
)

from image_similarity.similarity_data import (
    create_transform,
    get_image_paths,
)

from image_similarity.similarity_model import (
    ConvEncoder,
)


class SimilarityService:
    """
    相似商品检索服务。

    Flask 启动时完成：
        1. 加载编码器
        2. 加载 embeddings.npy
        3. 读取商品图片路径
        4. 创建 KNN 检索器
    """

    def __init__(self):

        # ==================== 1. 设置推理设备 ====================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "相似检索推理设备：",
            self.device,
        )


        # ==================== 2. 检查必要文件 ====================

        if not ENCODER_MODEL_PATH.exists():
            raise FileNotFoundError(
                "没有找到相似检索编码器参数："
                f"{ENCODER_MODEL_PATH}"
            )

        if not EMBEDDING_PATH.exists():
            raise FileNotFoundError(
                "没有找到商品特征向量库："
                f"{EMBEDDING_PATH}"
            )

        if not IMG_DIR.exists():
            raise FileNotFoundError(
                "没有找到商品图片目录："
                f"{IMG_DIR}"
            )


        # ==================== 3. 加载编码器 ====================

        # 创建与训练阶段完全一致的编码器结构
        self.encoder = ConvEncoder().to(
            self.device
        )

        # 从 encoder.pth 中读取训练好的模型参数
        encoder_state = torch.load(
            ENCODER_MODEL_PATH,
            map_location=self.device,
        )

        # 将参数安装到编码器
        self.encoder.load_state_dict(
            encoder_state
        )

        # 切换为推理模式
        self.encoder.eval()

        print(
            "相似检索编码器加载完成：",
            ENCODER_MODEL_PATH,
        )


        # ==================== 4. 加载商品特征库 ====================

        self.embeddings = np.load(
            EMBEDDING_PATH
        )

        # embeddings 应当是：
        # [商品图片数量, 特征维度]
        if self.embeddings.ndim != 2:
            raise RuntimeError(
                "商品特征库必须是二维矩阵，"
                f"当前形状为：{self.embeddings.shape}"
            )

        print(
            "商品特征库形状：",
            self.embeddings.shape,
        )


        # ==================== 5. 创建商品图片路径列表 ====================

        # 使用与生成 embeddings.npy 时相同的自然排序方式
        self.image_paths = get_image_paths(
            IMG_DIR
        )

        # 特征库的第 i 行必须对应图片列表中的第 i 张图片
        if (
            len(self.image_paths)
            != len(self.embeddings)
        ):
            raise RuntimeError(
                "商品图片数量与特征向量数量不一致："
                f"图片数量为 {len(self.image_paths)}，"
                f"特征数量为 {len(self.embeddings)}"
            )

        print(
            "商品图片数量：",
            len(self.image_paths),
        )


        # ==================== 6. 创建图片预处理 ====================

        # 网页上传图片必须使用与训练阶段相同的预处理
        self.transform = create_transform()


        # ==================== 7. 创建 KNN 检索器 ====================

        self.knn = NearestNeighbors(
            n_neighbors=NUM_SIMILAR_IMAGES,
            metric="cosine",
        )

        # 将完整商品特征库登记到 KNN 中
        # 这不是神经网络训练，不会更新编码器参数
        self.knn.fit(
            self.embeddings
        )

        print(
            "相似商品 KNN 检索器创建完成"
        )


    def extract_feature(
            self,
            image,
    ):
        """
        从一张 PIL 图片中提取 1024 维特征。

        输入：
            image：PIL.Image

        返回：
            feature_vector：
            形状为 [1, 1024] 的 NumPy 数组
        """

        if not isinstance(
                image,
                Image.Image,
        ):
            raise TypeError(
                "image 必须是 PIL.Image 类型"
            )


        # ==================== 1. 转换为 RGB ====================

        image = image.convert(
            "RGB"
        )


        # ==================== 2. 图片预处理 ====================

        # [3, 64, 64]
        image_tensor = self.transform(
            image
        )

        # 增加 batch 维度
        # [3, 64, 64]
        # →
        # [1, 3, 64, 64]
        image_batch = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )


        # ==================== 3. 编码器提取特征 ====================

        with torch.inference_mode():

            # [1, 3, 64, 64]
            # →
            # [1, 256, 2, 2]
            encoded_features = self.encoder(
                image_batch
            )

            # [1, 256, 2, 2]
            # →
            # [1, 1024]
            feature_vector = torch.flatten(
                encoded_features,
                start_dim=1,
            )


        # KNN 使用 NumPy 数组
        feature_vector = (
            feature_vector
            .cpu()
            .numpy()
        )


        # 检查查询特征维度是否与特征库一致
        if (
            feature_vector.shape[1]
            != self.embeddings.shape[1]
        ):
            raise RuntimeError(
                "查询图片特征维度与商品特征库不一致："
                f"查询维度为 {feature_vector.shape[1]}，"
                f"特征库维度为 {self.embeddings.shape[1]}"
            )

        return feature_vector


    def search_pil_image(
            self,
            image,
            num_images=NUM_SIMILAR_IMAGES,
    ):
        """
        检索与一张 PIL 图片最相似的商品图片。

        返回：
            一个列表，每个元素包含：
            排名、图片索引、文件名、余弦距离和相似度。
        """

        if num_images <= 0:
            raise ValueError(
                "num_images 必须大于 0"
            )

        if num_images > len(
                self.embeddings
        ):
            raise ValueError(
                "返回图片数量不能超过商品库数量"
            )


        # ==================== 1. 提取查询图片特征 ====================

        feature_vector = self.extract_feature(
            image
        )


        # ==================== 2. 执行 KNN 查询 ====================

        distances, indices = (
            self.knn.kneighbors(
                feature_vector,
                n_neighbors=num_images,
            )
        )

        # 当前只查询一张图片
        # 因此取第 0 行结果
        distances = distances[0]
        indices = indices[0]


        # ==================== 3. 整理查询结果 ====================

        search_results = []

        for rank, (
                image_index,
                distance,
        ) in enumerate(
            zip(
                indices,
                distances,
            ),
            start=1,
        ):
            image_index = int(
                image_index
            )

            distance = float(
                distance
            )

            image_path = (
                self.image_paths[
                    image_index
                ]
            )

            # 余弦相似度约等于：
            # 1 - 余弦距离
            similarity_value = (
                1.0 - distance
            )

            # 将相似度限制在 0～1
            similarity_value = max(
                0.0,
                min(
                    1.0,
                    similarity_value,
                ),
            )

            search_results.append({
                "rank": rank,

                "image_index": (
                    image_index
                ),

                "filename": (
                    image_path.name
                ),

                "distance": round(
                    distance,
                    6,
                ),

                "similarity_percent": round(
                    similarity_value * 100,
                    2,
                ),
            })

        return search_results


    def search_image_path(
            self,
            image_path,
            num_images=NUM_SIMILAR_IMAGES,
    ):
        """
        根据图片文件路径执行相似商品检索。
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"没有找到查询图片：{image_path}"
            )

        with Image.open(
            image_path
        ) as image:

            search_results = (
                self.search_pil_image(
                    image=image,
                    num_images=num_images,
                )
            )

        return search_results


# ==================== 创建全局服务对象 ====================

# Flask 启动时只加载一次编码器、特征库和 KNN
similarity_service = SimilarityService()


# ==================== 单独运行烟雾测试 ====================

if __name__ == "__main__":

    test_image_path = (
        IMG_DIR
        / "0.jpg"
    )

    print(
        "相似检索测试图片：",
        test_image_path,
    )

    test_results = (
        similarity_service
        .search_image_path(
            image_path=test_image_path,
            num_images=NUM_SIMILAR_IMAGES,
        )
    )

    print(
        "相似商品检索结果："
    )

    for result in test_results:

        print(
            f"第 {result['rank']} 名，"
            f"索引：{result['image_index']}，"
            f"文件：{result['filename']}，"
            f"余弦距离：{result['distance']:.6f}，"
            f"特征相似度："
            f"{result['similarity_percent']:.2f}%"
        )