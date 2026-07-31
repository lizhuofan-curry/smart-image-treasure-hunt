# 本文件负责加载商品分类模型，并对单张外部图片进行分类预测

from pathlib import Path

from PIL import Image
import torch

from image_classification.classification_config import (
    CLASSIFIER_MODEL_PATH,  # 分类模型参数完整保存路径
    CLASSIFICATION_NAMES,   # 将分类标签映射为中文类别
    IMG_DIR                 # 数据集图片地址
)

from image_classification.classification_data import create_transform

from image_classification.classification_model import Classifier

# 商品分类推理服务
# 模型只在创建服务对象时加载一次
class ClassificationService:
    def __init__(self):
        # 设置推理设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('商品分类推理设备：',self.device)

        # 检查模型参数
        if not CLASSIFIER_MODEL_PATH.exists():
            raise FileNotFoundError('没有找到商品分类模型参数：'f'{CLASSIFIER_MODEL_PATH}')

        # 创建并加载模型
        # 创建和训练阶段完全一致的模型结构
        self.model = Classifier().to(self.device)

        # 从 classifier.pth 读取训好的参数
        model_state = torch.load(CLASSIFIER_MODEL_PATH,self.device)
        self.model.load_state_dict(model_state)  # 参数装到模型

        # 切换到推理模式，batchnorm 使用训练阶段保存的数据
        self.model.eval()

        # 创建图片预处理
        self.transform = create_transform()
    '''
    对一张PIL图片进行商品分类
    输入： image:PIL.Image 图片
    返回： 商品类别编号，类别名称和置信度
    '''
    def predict_pil_image(
            self,
            image
    ):
        if not isinstance(image, Image.Image):  # todo
            raise TypeError('image 必须是 PIL.Image 类型')

        # 统一图片格式
        image = image.convert('RGB')

        # 图片预处理
        image_tensor = self.transform(image)

        # 增加batch维度，方便送进model里面向前传播
        image_batch = image_tensor.unsqueeze(0).to(self.device)

        # 模型推理,推理阶段不计算梯度
        with torch.inference_mode():    # todo

            # output 形状为 [1,5]
            # 里面是五个类别的原始预测分数
            outputs =self.model(image_batch)

            # 将五个原始分数转化为概率
            probabilities = torch.softmax(outputs,1)

            # 找到最高概率以及其类别编号
            confidence,predicted = torch.max(probabilities,1)

            # 整理预测结果
            class_id = int(predicted.item())
            confidence_value = float(confidence.item())
            class_name = CLASSIFICATION_NAMES[class_id]

            return {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence_value, # 0~1
                'confidence_percent' : round(100 * confidence_value, 2),  # 百分比 todo
            }

    # 根据图片路径打开图片并进行分类
    def predict_image_path(self,image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f'没有找到待分类图片:{image_path}')

        with Image.open(image_path) as image:
            result = self.predict_pil_image(image)

        return result

# 创建全局分类服务对象
# 以后 web_app.py 导入时，模型只加载一次
# Flask 启动 -> 加载classifier.pth 一次 -> 对上传的照片直接推理
classifier_service = ClassificationService()

# 单独运行当前文件时进行烟雾测试
if __name__ == "__main__":

    test_image_path = (
        IMG_DIR
        / "0.jpg"
    )

    print(
        "测试图片：",
        test_image_path,
    )

    test_result = (
        classifier_service
        .predict_image_path(
            test_image_path
        )
    )

    print(
        "分类预测结果：",
        test_result,
    )