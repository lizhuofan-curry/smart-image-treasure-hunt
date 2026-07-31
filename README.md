# 智图寻宝 · Shop Vision

> 一个将商品图像分类、图像去噪与相似商品检索整合到同一 Flask Web 界面的计算机视觉实践项目。

## 项目简介

“智图寻宝”面向商品图片场景，将三条独立的 PyTorch 推理链路接入统一网页：上传一张图片后，可获得商品类别预测、加噪/去噪对比结果，或在商品图库中检索最相近的 5 张图片。项目的重点不仅是训练单个模型，也包括模型权重、特征库、图片数据与 Web 服务之间的工程衔接。

## 功能亮点

- **商品分类**：四个卷积-批归一化-池化模块逐步提取特征（16 → 32 → 64 → 128 通道），通过全局平均池化输出上衣、鞋、包、下身衣服、手表共 5 类预测。
- **图像去噪**：使用带跳跃连接的轻量 U-Net 风格卷积自编码器；编码器保留多尺度纹理，解码器将对应特征拼接回来，以恢复 64 × 64 RGB 图像。
- **相似商品检索**：五层卷积编码器将图片编码为 1024 维特征，基于预计算特征库使用余弦距离 KNN 返回 Top-5 相似商品，并自动过滤查询图片自身。
- **可交互 Web 应用**：Flask 统一管理上传、推理和结果展示；限制图片格式为 JPG/JPEG/PNG/BMP，单文件上限 10 MB，上传文件使用 UUID 避免重名覆盖。
- **可复现训练设置**：公共工具函数固定 Python、NumPy、PyTorch 与 cuDNN 随机性；各模块将数据路径、随机种子、批大小和学习率集中配置。

## 项目结构

```text
.
├── common/                    # 公共工具、标签文件和本地数据目录
├── image_classification/      # 5 类商品分类：数据、模型、训练与测试
├── image_denoising/           # U-Net 风格去噪：数据、模型、训练与测试
├── image_similarity/          # 自编码器特征、嵌入库与 KNN 检索
├── web/                       # Flask 入口、服务层、模板和静态资源
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 创建环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如使用 GPU，请按你的 CUDA 环境从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应的 `torch` 与 `torchvision` 版本。

### 2. 准备本地资源

为避免将约数百 MB 的图片数据和模型产物直接提交到 Git，本仓库忽略下列本地文件。请按项目原始资料恢复到对应位置：

```text
common/dataset/                         # 商品图片库
image_classification/classifier.pth     # 分类模型权重
image_denoising/denoiser.pth            # 去噪模型权重
image_similarity/encoder.pth            # 检索编码器权重
image_similarity/decoder.pth            # 检索解码器权重
image_similarity/embeddings.npy          # 与图片自然排序一一对应的特征库
```

`common/fashion-labels.csv` 保留在仓库中，用于商品分类标签映射。相似检索要求 `embeddings.npy` 的行数与 `common/dataset/` 中按自然排序得到的图片数量完全一致。

### 3. 启动 Web 服务

在项目根目录运行：

```bash
python -m web.web_app
```

浏览器访问 <http://127.0.0.1:9000>，上传商品图片后选择“商品分类”“图像去噪”或“相似商品”。

## 模块说明

| 模块 | 模型与方法 | 输出 |
| --- | --- | --- |
| 商品分类 | CNN + Cross Entropy | 5 个商品类别中的预测结果 |
| 图像去噪 | 带跳跃连接的卷积编码器-解码器 + MSE | 加噪图片及其去噪结果 |
| 相似商品 | 卷积自编码器编码器 + 余弦距离 `NearestNeighbors` | Top-5 相似商品、距离与相似度 |

## 训练与推理

各模块均提供独立的 `*_train.py` 与 `*_test.py`。训练前请准备上述数据集；模型训练完成后会产生权重文件，检索模块还会生成 `embeddings.npy`。网页启动时加载这三类模块所需的权重和特征库，因此缺少文件时会给出明确的路径错误。

## 技术栈

Python · PyTorch · Torchvision · Flask · Pillow · NumPy · Pandas · scikit-learn · HTML/CSS/JavaScript

## 说明

本仓库聚焦于课程/考核项目的代码与工程实现。README 对模型、损失函数和检索方法的描述均以当前代码为准；未在此处声称未经记录的精度、检索延迟或额外训练技巧。
