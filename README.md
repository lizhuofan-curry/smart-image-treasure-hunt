# ✨ 智图寻宝 · Shop Vision

<p align="center">
  <img src="https://raw.githubusercontent.com/lizhuofan-curry/smart-image-treasure-hunt/main/web/static/images/school_logo.png" alt="河南大学软件学院项目 Logo" width="140" />
</p>

<p align="center">
  <a href="https://github.com/lizhuofan-curry/smart-image-treasure-hunt/actions/workflows/ci.yml"><img src="https://github.com/lizhuofan-curry/smart-image-treasure-hunt/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Flask-Web%20Application-000000?logo=flask&amp;logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Git%20LFS-Assets-F05032?logo=gitlfs&amp;logoColor=white" alt="Git LFS" />
</p>

<p align="center"><b>🛍️ 商品图像分类 · ✨ 图像去噪 · 🔎 相似商品检索</b></p>

“智图寻宝”是一个面向商品图片场景的计算机视觉综合实践项目。它将三条独立的 PyTorch 推理链路接入同一个 Flask 网页：上传一张图片，即可完成商品类别判断、带噪图像恢复，或从商品图库中找出最相近的 5 件商品。

它关注的不只是“训练出一个模型”，还将**数据准备、模型权重、特征库、Web 推理与结果呈现**串成可复现的完整闭环：分类回答“它是什么”，去噪改善“看得是否清楚”，检索回答“与哪些商品相似”。

## 🏷️ 项目标签

`computer-vision` `deep-learning` `pytorch` `flask` `image-classification` `image-denoising` `image-retrieval` `autoencoder` `knn` `fashion`

## 🖥️ 网页界面

下图为本地 Flask 服务实际运行时的首页，提供统一上传入口，并在同一页面承载商品分类、图像去噪和相似商品检索。

<p align="center">
  <img src="docs/assets/web-home-full.png" alt="智图寻宝 Flask 网页首页实际运行截图" width="720" />
</p>

<p align="center"><sub>Flask 网页首页：图片上传区、功能入口与实时服务状态</sub></p>

## 💡 核心亮点

| 从算法到应用 | 项目中的落地方式 |
| --- | --- |
| 一图三用 | 同一上传入口支持分类、去噪和相似检索三类视觉任务，无需切换脚本。 |
| 多尺度细节恢复 | 去噪网络在解码阶段拼接编码器特征，让局部纹理与高层语义共同参与图像恢复。 |
| 可检索的视觉表达 | 编码器将 64 × 64 RGB 图像映射到 1024 维向量，KNN 在已有特征库上完成相似商品搜索。 |
| 可维护的工程结构 | 训练、测试、配置、推理服务按模块拆分，路径与超参数集中管理。 |
| 可复现的交付 | 随机种子工具、Git LFS 资产管理和 GitHub Actions CI 共同支撑复现与持续检查。 |

## 🧠 功能一览

| 模块 | 实现方式 | 用户可见结果 |
| --- | --- | --- |
| 商品分类 | 4 个卷积块（16 → 32 → 64 → 128）+ 全局平均池化 | 上衣、鞋、包、下身衣服、手表 5 类预测 |
| 图像去噪 | 带跳跃连接的轻量 U-Net 风格卷积自编码器 | 固定强度混合噪声下的去噪图像 |
| 相似商品检索 | 5 层卷积编码器、1024 维嵌入、余弦距离 KNN | Top-5 相似商品、距离与相似度 |
| Web 服务 | Flask + HTML/CSS/JavaScript | 上传、推理、结果展示的一体化页面 |

## 🔄 系统流程

```mermaid
flowchart LR
    A[上传商品图片] --> B[Flask 服务]
    B --> C[商品分类 CNN]
    B --> D[图像去噪 U-Net]
    B --> E[特征编码器]
    E --> F[余弦距离 KNN]
    C --> G[类别结果]
    D --> H[去噪图像]
    F --> I[Top-5 相似商品]
```

## 🗂️ 项目结构

```text
.
├── common/
│   ├── dataset.zip                 # 商品图片数据集（Git LFS）
│   ├── fashion-labels.csv          # 分类标签
│   └── utils.py                    # 随机种子与自然排序工具
├── image_classification/           # 分类数据、CNN、训练、测试、权重
├── image_denoising/                # 去噪数据、U-Net、训练、测试、权重
├── image_similarity/               # 自编码器、嵌入库、KNN 检索、权重
├── web/                            # Flask 入口、服务层、前端页面
├── docs/assets/                    # README 演示素材
├── .github/workflows/ci.yml        # 自动语法与项目结构检查
├── .gitattributes                  # Git LFS 规则
└── requirements.txt
```

## 🚀 快速开始

### 1. 克隆并获取大文件

数据集、模型权重和特征库通过 Git LFS 管理。请先安装 [Git LFS](https://git-lfs.com/)，再执行：

```bash
git clone https://github.com/lizhuofan-curry/smart-image-treasure-hunt.git
cd smart-image-treasure-hunt
git lfs pull
```

### 2. 解压数据集

```bash
python -c "import zipfile; zipfile.ZipFile('common/dataset.zip').extractall('common')"
```

解压后应存在 `common/dataset/`。相似检索的 `embeddings.npy` 与该目录按自然排序的图片一一对应，请勿替换或打乱图片顺序。

### 3. 安装 Python 依赖

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

若使用 GPU，请从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装与你的 CUDA 环境匹配的 `torch` 与 `torchvision`。

### 4. 启动应用

```bash
python -m web.web_app
```

打开 <http://127.0.0.1:9000>，上传 JPG、JPEG、PNG 或 BMP 图片（最大 10 MB），再选择“商品分类”“图像去噪”或“相似商品”。

### 🩺 启动前自检

若页面启动时报找不到文件，请按下表检查：

| 现象 | 优先检查 |
| --- | --- |
| 找不到图片目录 | 是否已将 `dataset.zip` 解压为 `common/dataset/` |
| 分类或去噪服务加载失败 | 对应的 `classifier.pth` 或 `denoiser.pth` 是否已被 Git LFS 拉取 |
| 相似检索加载失败 | `encoder.pth`、`embeddings.npy` 与图片数据集是否同时存在 |
| LFS 文件显示为文本指针 | 在仓库根目录执行 `git lfs install` 和 `git lfs pull` |

## 📦 模型与数据资产

| 文件 | 用途 | 版本管理 |
| --- | --- | --- |
| `common/dataset.zip` | 商品图片数据集压缩包 | Git LFS |
| `image_classification/classifier.pth` | 5 类分类模型 | Git LFS |
| `image_denoising/denoiser.pth` | 图像去噪模型 | Git LFS |
| `image_similarity/encoder.pth` | 相似检索特征编码器 | Git LFS |
| `image_similarity/decoder.pth` | 自编码器训练时的解码器 | Git LFS |
| `image_similarity/embeddings.npy` | 全量商品的 1024 维特征库 | Git LFS |

## 🛠️ 工程实践

- **输入安全**：仅接受图片格式，单文件限制为 10 MB；服务端保存时使用 UUID，避免同名覆盖。
- **可复现性**：固定 Python、NumPy、PyTorch 与 cuDNN 的随机性，并将训练超参数集中到各模块配置文件。
- **模型复用**：Flask 启动时加载推理服务；相似检索在启动阶段构建 KNN 索引，查询阶段直接复用。
- **持续集成**：每次推送至 `main` 或提交 Pull Request 时，GitHub Actions 会编译全部 Python 模块并核验关键项目文件。

## 🏆 项目成果与材料

- **可运行应用**：三个视觉模块已接入同一 Flask 页面，默认监听 `127.0.0.1:9000`。
- **可复现实验资产**：数据集压缩包、分类/去噪/检索模型权重与检索特征库均由 Git LFS 管理。
- **真实演示材料**：`docs/assets/web-home-full.png` 为本地 Flask 首页实际运行截图。
- **补充证明材料**：如需添加课程证书、获奖证明或项目答辩材料，请置于 `docs/certificates/`，并在本节追加来源和说明。

## 🗺️ 路线图

- [x] 三个 CV 模块的统一 Web 入口
- [x] Git LFS 大文件管理与基础 CI
- [x] README 网页界面展示与复现说明
- [ ] 增加自动化单元/集成测试与模型加载烟雾测试
- [ ] 容器化（Docker）并接入可选部署目标
- [ ] 配置 CD：镜像构建、部署及健康检查
- [ ] 补充更多真实的分类和相似检索演示样例

## 🔧 技术栈

Python · PyTorch · Torchvision · Flask · Pillow · NumPy · Pandas · scikit-learn · HTML/CSS/JavaScript · GitHub Actions · Git LFS

## 📌 说明

README 对模型结构、损失函数和检索方式均以仓库当前代码为准；网页截图用于说明交互界面，不应被解读为模型精度、检索延迟或泛化能力的量化结论。
