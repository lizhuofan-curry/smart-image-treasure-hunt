import time
import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API 配置文件
ENV_PATH = PROJECT_ROOT / "kimi.env"

# 加载环境变量
load_dotenv(ENV_PATH)


def image_to_data_url(image_path) -> str:
    """
    将本地图片转换成 Base64 Data URL。
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"没有找到图片：{image_path}"
        )

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        raise ValueError(
            "无法识别图片格式。"
        )

    if not mime_type.startswith("image/"):
        raise ValueError(
            "传入的文件不是图片。"
        )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        f"data:{mime_type};base64,{encoded_image}"
    )


def generate_product_description(image_path,) -> str:
    """
    根据商品图片生成中文商品介绍。
    """

    api_key = os.getenv("MOONSHOT_API_KEY")

    if not api_key:
        raise ValueError(
            "未读取到 MOONSHOT_API_KEY，"
            "请检查 kimi.env。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    image_url = image_to_data_url(image_path)

    user_prompt = """
    请仔细分析这张商品图片，并生成一段150字以内的中文商品介绍。

    介绍中可以包含：

    1. 图片中的商品类别；
    2. 商品的主要颜色；
    3. 商品的外观和造型特点；
    4. 可能适用的使用场景；
    5. 可能适用的人群。

    要求：

    1. 只描述图片中能够观察到的信息；
    2. 不得虚构品牌、价格和生产厂家；
    3. 不得虚构无法确认的具体材质；
    4. 不得编造无法确认的性能参数；
    5. 不确定的信息使用“可能”“看起来”等表达；
    6. 使用客观、简洁、自然的中文。
    """.strip()

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名严谨的商品图片分析助手。"
                    "只能根据用户上传的图片进行描述，"
                    "不得虚构图片中无法确认的信息。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ],

        # 关闭深度思考，商品介绍不需要复杂推理
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },

    )
    elapsed_time = (
            time.perf_counter()
            - start_time
    )

    print(
        f"Kimi API 耗时：{elapsed_time:.2f} 秒"
    )

    result = response.choices[0].message.content

    if not result:
        raise RuntimeError(
            "Kimi 返回了空内容。"
        )

    return result