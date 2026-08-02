# 本文件负责加载图像去噪模型，并生成加噪图片和模型去噪图片

from pathlib import Path
from uuid import uuid4

from PIL import Image
import torch
from torchvision import transforms as T

from image_denoising.denoising_config import (
    DENOISER_MODEL_PATH,
    IMG_PATH,
    SEED,
    TEST_GAUSSIAN_NOISE_FACTOR,
    TEST_SALT_PEPPER_RATIO,
)

from image_denoising.denoising_data import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    create_transform,
)

from image_denoising.denoising_model import (
    ConvDenoiser,
)


class DenoisingService:
    # 图像去噪推理服务
    # 模型只在创建服务对象时加载一次

    def __init__(self):

        # ==================== 1. 设置推理设备 ====================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "图像去噪推理设备：",
            self.device,
        )


        # ==================== 2. 检查模型参数文件 ====================

        if not DENOISER_MODEL_PATH.exists():
            raise FileNotFoundError(
                "没有找到图像去噪模型参数："
                f"{DENOISER_MODEL_PATH}"
            )


        # ==================== 3. 创建并加载去噪模型 ====================

        self.model = ConvDenoiser().to(
            self.device
        )

        model_state = torch.load(
            DENOISER_MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(
            model_state
        )

        # BatchNorm 使用训练阶段保存的统计数据
        self.model.eval()


        # ==================== 4. 创建图片转换工具 ====================

        # 与训练阶段完全相同的图片预处理
        self.transform = create_transform()

        # 将 [3, H, W] Tensor 转换回 PIL 图片
        self.to_pil_image = T.ToPILImage()

        print(
            "图像去噪模型加载完成：",
            DENOISER_MODEL_PATH,
        )


    def create_noisy_image(
            self,
            clean_image,
    ):
        """
        为清晰图片添加固定的混合噪声。

        输入：
            clean_image：[3, 64, 64]

        返回：
            noisy_image：[3, 64, 64]
        """

        if clean_image.ndim != 3:
            raise ValueError(
                "clean_image 必须是三维 Tensor："
                "[C, H, W]"
            )

        # 固定随机种子，使网页演示结果可复现
        torch.manual_seed(SEED)

        # 先添加高斯噪声
        gaussian_image = add_gaussian_noise(
            image=clean_image,
            noise_factor=(
                TEST_GAUSSIAN_NOISE_FACTOR
            ),
        )

        # 再叠加椒盐噪声
        noisy_image = add_salt_pepper_noise(
            image=gaussian_image,
            noise_ratio=(
                TEST_SALT_PEPPER_RATIO
            ),
        )

        return noisy_image


    def denoise_pil_image(
            self,
            image,
    ):
        """
        对一张 PIL 图片添加噪声并执行模型去噪。

        返回：
            clean_image：预处理后的原始图片
            noisy_image：添加混合噪声后的图片
            denoised_image：模型恢复后的图片
        """

        if not isinstance(
                image,
                Image.Image,
        ):
            raise TypeError(
                "image 必须是 PIL.Image 类型"
            )


        # ==================== 1. 统一为 RGB 图片 ====================

        image = image.convert("RGB")


        # ==================== 2. 执行统一预处理 ====================

        # [3, 64, 64]
        clean_image = self.transform(
            image
        )


        # ==================== 3. 添加固定混合噪声 ====================

        # [3, 64, 64]
        noisy_image = self.create_noisy_image(
            clean_image
        )


        # ==================== 4. 增加 batch 维度 ====================

        # [3, 64, 64]
        # →
        # [1, 3, 64, 64]
        noisy_batch = (
            noisy_image
            .unsqueeze(0)
            .to(self.device)
        )


        # ==================== 5. 模型推理 ====================

        with torch.inference_mode():

            # 模型输出的是预测噪声
            # [1, 3, 64, 64]
            predicted_noise_batch = self.model(
                noisy_batch
            )

            # 带噪图片 - 预测噪声 = 去噪图片
            denoised_batch = (noisy_batch - predicted_noise_batch)

            # 将像素限制在 [0,1] 范围内
            denoised_batch = torch.clamp(denoised_batch, 0.0, 1.0)


        # 去掉 batch 维度
        # [1, 3, 64, 64]
        # →
        # [3, 64, 64]
        denoised_image = (
            denoised_batch[0]
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
        )

        return (
            clean_image.cpu(),
            noisy_image.cpu(),
            denoised_image,
        )


    def process_image_path(
            self,
            image_path,
            output_dir,
    ):
        """
        根据图片路径执行去噪，并保存网页需要显示的结果。

        输入：
            image_path：已上传原始图片路径
            output_dir：结果图片保存目录

        返回：
            包含结果路径和噪声参数的字典
        """

        image_path = Path(
            image_path
        )

        output_dir = Path(
            output_dir
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"没有找到待处理图片：{image_path}"
            )

        # 自动创建结果目录
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ==================== 1. 打开并处理图片 ====================

        with Image.open(image_path) as image:

            (
                clean_image,
                noisy_image,
                denoised_image,
            ) = self.denoise_pil_image(
                image
            )


        # ==================== 2. 创建唯一结果文件名 ====================

        result_id = uuid4().hex

        noisy_filename = (
            f"noisy_{result_id}.png"
        )

        denoised_filename = (
            f"denoised_{result_id}.png"
        )

        noisy_path = (
            output_dir
            / noisy_filename
        )

        denoised_path = (
            output_dir
            / denoised_filename
        )


        # ==================== 3. 保存加噪和去噪图片 ====================

        noisy_pil_image = (
            self.to_pil_image(
                noisy_image.clamp(
                    0.0,
                    1.0,
                )
            )
        )

        denoised_pil_image = (
            self.to_pil_image(
                denoised_image
            )
        )

        noisy_pil_image.save(
            noisy_path
        )

        denoised_pil_image.save(
            denoised_path
        )


        # ==================== 4. 返回网页需要的信息 ====================

        return {
            "noisy_filename": (
                noisy_filename
            ),

            "denoised_filename": (
                denoised_filename
            ),

            "noisy_path": (
                noisy_path
            ),

            "denoised_path": (
                denoised_path
            ),

            "gaussian_noise_factor": (
                TEST_GAUSSIAN_NOISE_FACTOR
            ),

            "salt_pepper_ratio": (
                TEST_SALT_PEPPER_RATIO
            ),
        }


# 创建全局去噪服务
# Flask 启动时加载一次模型
denoising_service = DenoisingService()


# 单独运行当前文件时进行烟雾测试
if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        .parent
    )

    test_image_path = (
        IMG_PATH
        / "100.jpg"
    )

    test_output_dir = (
        project_root
        / "web"
        / "static"
        / "results"
        / "denoising"
    )

    print(
        "去噪测试图片：",
        test_image_path,
    )

    test_result = (
        denoising_service
        .process_image_path(
            image_path=test_image_path,
            output_dir=test_output_dir,
        )
    )

    print(
        "图像去噪结果：",
        test_result,
    )