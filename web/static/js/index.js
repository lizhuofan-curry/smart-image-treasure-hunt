// 本文件负责处理图片预览、按钮状态及分类和相似度进度条


// 获取页面元素
const imageInput =
    document.getElementById("imageInput");

const selectedFileName =
    document.getElementById("selectedFileName");

const previewImage =
    document.getElementById("previewImage");

const emptyPreview =
    document.getElementById("emptyPreview");

const uploadForm =
    document.getElementById("uploadForm");

const uploadButton =
    document.getElementById("uploadButton");


// 保存当前创建的临时图片地址
let currentPreviewUrl = null;


// 监听图片选择
if (
    imageInput
    && selectedFileName
    && previewImage
) {
    imageInput.addEventListener(
        "change",
        function () {

            // 取得用户选择的第一张图片
            const selectedFile =
                imageInput.files[0];


            // 用户取消选择
            if (!selectedFile) {
                selectedFileName.textContent =
                    "暂未选择图片";

                return;
            }


            // 检查文件 MIME 类型
            if (
                !selectedFile.type.startsWith(
                    "image/"
                )
            ) {
                selectedFileName.textContent =
                    "请选择有效的图片文件";

                imageInput.value = "";

                return;
            }


            // 显示文件名
            selectedFileName.textContent =
                "已选择：" + selectedFile.name;


            // 释放旧的临时预览地址
            if (currentPreviewUrl) {
                URL.revokeObjectURL(
                    currentPreviewUrl
                );
            }


            /*
                创建浏览器本地预览地址。
                这里只显示图片，还没有上传到 Flask。
            */
            currentPreviewUrl =
                URL.createObjectURL(
                    selectedFile
                );


            // 显示原始图片预览
            previewImage.src =
                currentPreviewUrl;

            previewImage.hidden =
                false;


            // 隐藏空白占位内容
            if (emptyPreview) {
                emptyPreview.hidden =
                    true;
            }
        }
    );
}


// 上传表单提交时禁用按钮
if (
    uploadForm
    && uploadButton
) {
    uploadForm.addEventListener(
        "submit",
        function () {

            uploadButton.disabled =
                true;

            uploadButton.textContent =
                "正在上传图片...";
        }
    );
}


// 页面关闭前释放临时地址
window.addEventListener(
    "beforeunload",
    function () {

        if (currentPreviewUrl) {
            URL.revokeObjectURL(
                currentPreviewUrl
            );
        }
    }
);

// 根据 Flask 返回的 Softmax 概率设置进度条宽度
const confidenceFill =
    document.querySelector(
        ".confidence-fill"
    );

if (confidenceFill) {

    const confidenceValue =
        Number(
            confidenceFill.dataset.confidence
        );

    // 保证进度值处于 0 到 100 之间
    const safeConfidenceValue =
        Number.isFinite(confidenceValue)
            ? Math.min(
                100,
                Math.max(
                    0,
                    confidenceValue
                )
            )
            : 0;

    confidenceFill.style.width =
        safeConfidenceValue + "%";
}
// 根据相似检索结果设置每张卡片的特征相似度进度条
const similarityFills =
    document.querySelectorAll(
        ".similarity-fill"
    );

similarityFills.forEach(
    function (similarityFill) {

        const similarityValue =
            Number(
                similarityFill.dataset.similarity
            );

        const safeSimilarityValue =
            Number.isFinite(similarityValue)
                ? Math.min(
                    100,
                    Math.max(
                        0,
                        similarityValue
                    )
                )
                : 0;

        similarityFill.style.width =
            safeSimilarityValue + "%";
    }
);


// 运行模型时禁用当前模块按钮，防止重复提交
const moduleForms =
    document.querySelectorAll(
        ".module-form"
    );

moduleForms.forEach(
    function (moduleForm) {

        moduleForm.addEventListener(
            "submit",
            function () {

                const moduleButton =
                    moduleForm.querySelector(
                        ".module-card"
                    );

                const moduleState =
                    moduleForm.querySelector(
                        ".module-state"
                    );

                if (moduleButton) {
                    moduleButton.disabled =
                        true;
                }

                if (moduleState) {
                    moduleState.textContent =
                        "正在运行...";
                }
            }
        );
    }
);
