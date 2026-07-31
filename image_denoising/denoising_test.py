# 本文件负责加载训练完成的去噪模型，并可视化模型的去噪效果

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common.utils import seed_everything
# 导入最佳参数
from image_denoising.denoising_config import DENOISER_MODEL_PATH,SEED,TEST_SALT_PEPPER_RATIO,TEST_GAUSSIAN_NOISE_FACTOR
from image_denoising.denoising_data import create_dataset,add_gaussian_noise,add_salt_pepper_noise
from image_denoising.denoising_model import ConvDenoiser

# 加载验证损失最低的模型参数
# 并展示带噪图片，模型输出和清晰目标图片
def test_model():
    # 固定随机种子
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前测试设备：', device)

    # create_dataset() 返回训练集和验证集
    # 测试时，只使用验证集
    _,val_dataset = create_dataset()

    # 测试阶段不需要打乱图片顺序
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size = 4,
        shuffle = False,
        num_workers = 0,
    )

    # 创建与训练阶段完全相同的模型结构
    model = ConvDenoiser().to(device)

    # 加载最佳模型参数
    model_state  =  torch.load(
        DENOISER_MODEL_PATH,
        map_location=device
    )
    model.load_state_dict(model_state)

    # 切换为验证模式
    model.eval()

    # val_dataset 原本返回随机带噪图片和清晰图片
    # 测试时忽略随机带噪图片，用清晰图片生成固定带噪
    _,clean_images = next(iter(val_loader))
    clean_images = clean_images.to(device)

    # 重新固定随机状态，保证每次测试使用相同的噪声图案
    seed_everything(SEED)

    # 使用config 中固定的参数，为每张图片生成相同标准的混合噪声
    noisy_image_list =[]
    for clean_image in clean_images:
        gaussian_image = add_gaussian_noise(
            clean_image,
            TEST_GAUSSIAN_NOISE_FACTOR,
        )
        mixed_noisy_image = add_salt_pepper_noise(
            gaussian_image,
            TEST_SALT_PEPPER_RATIO,
        )
        noisy_image_list.append(mixed_noisy_image)

    # torch.stack 是把图片重新组合成 batch
    noisy_images = torch.stack(noisy_image_list)
    noisy_images = noisy_images.to(device)
    # 测试阶段不计算梯度
    with torch.no_grad():
        denoised_images = model(noisy_images)

    print("带噪图片形状：", noisy_images.shape)
    print("去噪图片形状：", denoised_images.shape)
    print("清晰图片形状：", clean_images.shape)

    # 将图片移回 CPU，供 matplotlib 显示
    noisy_images = noisy_images.cpu()
    denoised_images = denoised_images.cpu()
    clean_images = clean_images.cpu()

    # 每一行显示一张样本
    # 第一列是带噪图片，第二列是模型输出，第三列是清晰目标
    plt.figure(figsize=(9,12))

    for index in range(4):
        # 显示带噪图片
        plt.subplot(4,3,index*3+1)
        plt.imshow(noisy_images[index].permute(1,2,0).numpy())
        plt.title('Noisy')
        plt.axis('off')

        # 显示模型恢复图片
        plt.subplot(4,3,index*3+2)
        plt.imshow(denoised_images[index].permute(1,2,0).numpy())
        plt.title('Denosied')
        plt.axis('off')

        # 显示原始清晰图片
        plt.subplot(4,3,index*3+3)
        plt.imshow(clean_images[index].permute(1,2,0).numpy())
        plt.title('Clean')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model()

