# 本文件负责启动 Flask 服务、保存上传图片，并调用分类、去噪和相似检索模型

from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)

from image_similarity.similarity_config import (
    IMG_DIR,
    NUM_SIMILAR_IMAGES,
)

from web.services.classification_service import (
    classifier_service,
)

from web.services.denoising_service import (
    denoising_service,
)

from web.services.similarity_service import (
    similarity_service,
)

from common.llm_service import (
    generate_product_description,
)
# ==================== 1. 路径配置 ====================

# 当前 web 模块目录
WEB_DIR = Path(__file__).resolve().parent

# 用户上传的原始图片目录
UPLOAD_DIR = (
    WEB_DIR
    / "static"
    / "uploads"
)

# 去噪结果保存目录
DENOISING_RESULT_DIR = (
    WEB_DIR
    / "static"
    / "results"
    / "denoising"
)

# 自动创建目录
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DENOISING_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# 允许上传的图片后缀
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ==================== 2. 创建 Flask 应用 ====================

app = Flask(__name__)

# 上传文件最大限制为 10MB
app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ==================== 3. 获取上传图片路径 ====================

def get_uploaded_image_path(
        uploaded_filename,
):
    """
    根据服务器文件名取得上传图片的完整路径。
    """

    if not uploaded_filename:
        raise ValueError(
            "没有收到已上传图片的文件名"
        )

    # 只保留文件名，防止访问其他目录
    safe_filename = Path(
        uploaded_filename
    ).name

    image_path = (
        UPLOAD_DIR
        / safe_filename
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"没有找到已上传图片：{image_path}"
        )

    return image_path


# ==================== 4. 数据集图片访问路由 ====================
@app.route("/ping")
def ping():
    print(
        ">>> 浏览器已经连接到当前 Flask 程序",
        flush=True,
    )

    return "当前 web_app.py 连接成功"

@app.route(
    "/dataset/<path:filename>",
    methods=["GET"],
)
def dataset_image(
        filename,
):
    """
    安全地向浏览器返回 common/dataset 中的商品图片。
    """

    safe_filename = Path(
        filename
    ).name

    return send_from_directory(
        directory=IMG_DIR,
        path=safe_filename,
    )


# ==================== 5. 首页路由 ====================

@app.route(
    "/",
    methods=["GET", "POST"],
)
def index():

    # 上传到服务器后的 UUID 文件名
    uploaded_filename = None

    # 用户电脑中的原始文件名
    original_filename = None

    # 分类结果
    prediction = None

    # 去噪结果
    denoising_result = None

    # 去噪流程中的三组分类结果
    original_prediction = None
    noisy_prediction = None
    denoised_prediction = None

    # 相似检索结果
    similarity_results = None

    # AI 商品介绍
    ai_description = None

    # 当前运行的模块
    active_module = None

    # 错误信息
    error_message = None


    # ==================== 6. 处理 POST 请求 ====================

    if request.method == "POST":

        # 获取用户点击的操作
        action = request.form.get(
            "action",
            "",
        )
        print(
            "收到的表单：",
            request.form.to_dict(),
        )

        print(
            "收到的 action：",
            repr(action),
        )

        # ==================== 7. 上传图片 ====================

        if action == "upload":

            uploaded_file = (
                request.files.get(
                    "image"
                )
            )

            if (
                uploaded_file is None
                or not uploaded_file.filename
            ):
                error_message = (
                    "请先选择一张商品图片"
                )

            else:
                original_filename = (
                    uploaded_file.filename
                )

                file_extension = (
                    Path(original_filename)
                    .suffix
                    .lower()
                )

                if (
                    file_extension
                    not in ALLOWED_EXTENSIONS
                ):
                    error_message = (
                        "不支持该文件格式，"
                        "请上传 JPG、JPEG、PNG 或 BMP 图片"
                    )

                else:
                    # 使用 UUID，避免同名图片覆盖
                    uploaded_filename = (
                        f"{uuid4().hex}"
                        f"{file_extension}"
                    )

                    save_path = (
                        UPLOAD_DIR
                        / uploaded_filename
                    )

                    try:
                        # 上传阶段只保存图片
                        uploaded_file.save(
                            save_path
                        )

                        print(
                            "图片上传完成：",
                            save_path,
                        )

                    except OSError as error:
                        error_message = (
                            f"图片保存失败：{error}"
                        )

                        if save_path.exists():
                            save_path.unlink()

                        uploaded_filename = None


        # ==================== 8. 商品分类 ====================

        elif action == "classify":

            uploaded_filename = (
                request.form.get(
                    "uploaded_filename",
                    "",
                )
            )

            original_filename = (
                request.form.get(
                    "original_filename",
                    "",
                )
            )

            try:
                image_path = (
                    get_uploaded_image_path(
                        uploaded_filename
                    )
                )

                prediction = (
                    classifier_service
                    .predict_image_path(
                        image_path
                    )
                )

                active_module = (
                    "classification"
                )

                print(
                    "商品分类图片：",
                    image_path,
                )

                print(
                    "商品分类结果：",
                    prediction,
                )

            except (
                    FileNotFoundError,
                    ValueError,
                    OSError,
                    TypeError,
                    RuntimeError,
            ) as error:

                error_message = (
                    f"商品分类失败：{error}"
                )


        # ==================== 9. 图像去噪 ====================

        elif action == "denoise":

            uploaded_filename = (
                request.form.get(
                    "uploaded_filename",
                    "",
                )
            )

            original_filename = (
                request.form.get(
                    "original_filename",
                    "",
                )
            )

            try:
                # 找到已经上传的原始图片
                image_path = (
                    get_uploaded_image_path(
                        uploaded_filename
                    )
                )

                # 添加噪声并调用 U-Net 去噪
                denoising_result = (
                    denoising_service
                    .process_image_path(
                        image_path=image_path,
                        output_dir=(
                            DENOISING_RESULT_DIR
                        ),
                    )
                )

                # 分别对原图、加噪图和去噪图进行分类，
                # 用于直观展示噪声对分类的影响以及去噪后的恢复效果。
                original_prediction = (
                    classifier_service
                    .predict_image_path(
                        image_path
                    )
                )

                noisy_prediction = (
                    classifier_service
                    .predict_image_path(
                        denoising_result[
                            "noisy_path"
                        ]
                    )
                )

                denoised_prediction = (
                    classifier_service
                    .predict_image_path(
                        denoising_result[
                            "denoised_path"
                        ]
                    )
                )

                active_module = "denoising"

                print(
                    "图像去噪输入：",
                    image_path,
                )

                print(
                    "图像去噪结果：",
                    denoising_result,
                )

                print(
                    "原图、加噪图、去噪图分类对比：",
                    {
                        "original": original_prediction,
                        "noisy": noisy_prediction,
                        "denoised": denoised_prediction,
                    },
                )

            except (
                    FileNotFoundError,
                    ValueError,
                    OSError,
                    TypeError,
                    RuntimeError,
            ) as error:

                error_message = (
                    f"图像去噪失败：{error}"
                )


        # ==================== 10. 相似商品检索 ====================

        elif action == "similarity":

            uploaded_filename = (
                request.form.get(
                    "uploaded_filename",
                    "",
                )
            )

            original_filename = (
                request.form.get(
                    "original_filename",
                    "",
                )
            )

            try:
                image_path = (
                    get_uploaded_image_path(
                        uploaded_filename
                    )
                )

                # 先判断查询图片的商品类别，
                # 再把相似检索限制在同一类别中。
                prediction = (
                    classifier_service
                    .predict_image_path(
                        image_path
                    )
                )

                search_class_id = (
                    prediction["class_id"]
                )

                # 多查询一些近邻。
                # 当上传图片本身来自商品库时，
                # 后面会排除距离接近 0 的同一张图片，
                # 仍然保证网页显示 5 张其他相似商品。
                search_count = min(
                    NUM_SIMILAR_IMAGES + 10,
                    len(
                        similarity_service
                        .class_image_indices[
                            search_class_id
                        ]
                    ),
                )

                raw_results = (
                    similarity_service
                    .search_image_path(
                        image_path=image_path,
                        num_images=search_count,
                        class_id=search_class_id,
                    )
                )

                # 排除与查询图完全相同的结果
                filtered_results = [
                    result
                    for result in raw_results
                    if result["distance"] > 1e-8
                ]

                # 正常情况下取排除自身后的前 5 张
                similarity_results = (
                    filtered_results[
                        :NUM_SIMILAR_IMAGES
                    ]
                )

                # 极端情况下全部都是零距离副本时，
                # 回退为原始前 5 个结果，避免页面为空
                if not similarity_results:
                    similarity_results = (
                        raw_results[
                            :NUM_SIMILAR_IMAGES
                        ]
                    )

                # 重新整理网页排名为 1～5
                for rank, result in enumerate(
                        similarity_results,
                        start=1,
                ):
                    result["rank"] = rank

                active_module = "similarity"

                print(
                    "相似检索输入：",
                    image_path,
                )

                print(
                    "相似检索类别：",
                    prediction,
                )

                print(
                    "相似检索结果：",
                    similarity_results,
                )

            except (
                    FileNotFoundError,
                    ValueError,
                    OSError,
                    TypeError,
                    RuntimeError,
            ) as error:

                error_message = (
                    f"相似检索失败：{error}"
                )


        # ==================== 11. AI 商品介绍 ====================

        elif action == "describe":

            uploaded_filename = (
                request.form.get(
                    "uploaded_filename",
                    "",
                )
            )

            original_filename = (
                request.form.get(
                    "original_filename",
                    "",
                )
            )

            try:
                # 找到用户已经上传的原始图片
                image_path = (
                    get_uploaded_image_path(
                        uploaded_filename
                    )
                )

                # 调用 Kimi 多模态模型生成商品介绍
                ai_description = (
                    generate_product_description(
                        image_path=image_path
                    )
                )

                # 告诉前端当前显示 AI 商品介绍模块
                active_module = "description"

                print(
                    "AI 商品介绍输入：",
                    image_path,
                )

                print(
                    "AI 商品介绍结果：",
                    ai_description,
                )

            except Exception as error:
                error_message = (
                    f"AI 商品介绍生成失败：{error}"
                )



        else:
            error_message = (
                "无法识别当前操作"
            )


    # ==================== 12. 渲染页面 ====================

    return render_template(
        "index.html",

        uploaded_filename=(
            uploaded_filename
        ),

        original_filename=(
            original_filename
        ),

        prediction=prediction,

        denoising_result=(
            denoising_result
        ),

        original_prediction=(
            original_prediction
        ),

        noisy_prediction=(
            noisy_prediction
        ),

        denoised_prediction=(
            denoised_prediction
        ),

        similarity_results=(
            similarity_results
        ),

        ai_description=(
            ai_description
        ),

        active_module=(
            active_module
        ),

        error_message=(
            error_message
        ),
    )


# ==================== 13. 启动 Flask ====================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=9001,
        debug=True,

        # 防止三个模型重复加载
        use_reloader=False,
    )
