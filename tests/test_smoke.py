import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from image_classification.classification_model import Classifier
from image_classification.classification_train import calculate_class_weights
from image_denoising.denoising_model import ConvDenoiser
from image_similarity.similarity_model import ConvDecoder, ConvEncoder


class ModelSmokeTests(unittest.TestCase):
    def test_class_weights_are_inverse_to_class_frequency(self):
        dataset = types.SimpleNamespace(
            labels_dict={
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 2,
                7: 3,
                8: 4,
                9: 4,
            }
        )
        train_subset = types.SimpleNamespace(
            dataset=dataset,
            indices=list(range(10)),
        )

        counts, weights = calculate_class_weights(train_subset)

        self.assertEqual(counts.tolist(), [4, 2, 1, 1, 2])
        self.assertTrue(
            torch.allclose(
                weights,
                torch.tensor([0.5, 1.0, 2.0, 2.0, 1.0]),
            )
        )

    def test_model_forward_shapes(self):
        images = torch.rand(2, 3, 64, 64)

        with torch.inference_mode():
            classification = Classifier().eval()(images)
            predicted_noise = ConvDenoiser().eval()(images)
            encoded = ConvEncoder().eval()(images)
            reconstructed = ConvDecoder().eval()(encoded)

        self.assertEqual(tuple(classification.shape), (2, 5))
        self.assertEqual(tuple(predicted_noise.shape), (2, 3, 64, 64))
        self.assertEqual(tuple(encoded.shape), (2, 256, 2, 2))
        self.assertEqual(tuple(reconstructed.shape), (2, 3, 64, 64))

    def test_state_dict_round_trip(self):
        source_model = Classifier()

        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = Path(temp_dir) / "classifier.pth"
            torch.save(source_model.state_dict(), weight_path)

            restored_model = Classifier()
            restored_model.load_state_dict(
                torch.load(
                    weight_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        self.assertEqual(
            source_model.classifier.out_features,
            restored_model.classifier.out_features,
        )


class RetrievalSmokeTests(unittest.TestCase):
    def test_knn_accepts_1024_dimension_embeddings(self):
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(12, 1024)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings[:1])

        self.assertEqual(distances.shape, (1, 5))
        self.assertEqual(indices.shape, (1, 5))
        self.assertEqual(int(indices[0, 0]), 0)


class FlaskSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeClassifierService:
            def predict_image_path(self, image_path):
                return {
                    "class_id": 3,
                    "class_name": "下身衣服",
                    "confidence": 0.96,
                    "confidence_percent": 96.0,
                }

        class FakeSimilarityService:
            def __init__(self):
                self.class_image_indices = {
                    3: np.arange(20),
                }
                self.last_class_id = None

            def search_image_path(
                    self,
                    image_path,
                    num_images,
                    class_id=None,
            ):
                self.last_class_id = class_id
                return [
                    {
                        "rank": index + 1,
                        "image_index": index,
                        "filename": f"{index}.jpg",
                        "class_id": 3,
                        "class_name": "下身衣服",
                        "distance": 0.1 + index * 0.01,
                        "similarity_percent": 90.0 - index,
                    }
                    for index in range(num_images)
                ]

        classification_module = types.ModuleType(
            "web.services.classification_service"
        )
        classification_module.classifier_service = (
            FakeClassifierService()
        )

        denoising_module = types.ModuleType(
            "web.services.denoising_service"
        )
        denoising_module.denoising_service = object()

        similarity_module = types.ModuleType(
            "web.services.similarity_service"
        )
        similarity_module.similarity_service = (
            FakeSimilarityService()
        )

        llm_module = types.ModuleType("common.llm_service")
        llm_module.generate_product_description = lambda image_path: "测试描述"

        sys.modules[classification_module.__name__] = classification_module
        sys.modules[denoising_module.__name__] = denoising_module
        sys.modules[similarity_module.__name__] = similarity_module
        sys.modules[llm_module.__name__] = llm_module

        from web import web_app

        cls.web_app = web_app
        cls.client = web_app.app.test_client()

    def test_similarity_route_uses_predicted_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.web_app.UPLOAD_DIR = Path(temp_dir)
            uploaded_path = Path(temp_dir) / "query.jpg"
            uploaded_path.write_bytes(b"fake-image")

            response = self.client.post(
                "/",
                data={
                    "action": "similarity",
                    "uploaded_filename": "query.jpg",
                    "original_filename": "query.jpg",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.web_app.similarity_service.last_class_id,
            3,
        )
        self.assertIn(
            "检索类别：下身衣服",
            response.get_data(as_text=True),
        )

    def test_upload_route_accepts_supported_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.web_app.UPLOAD_DIR = Path(temp_dir)
            response = self.client.post(
                "/",
                data={
                    "action": "upload",
                    "image": (io.BytesIO(b"smoke-test-image"), "sample.jpg"),
                },
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(list(Path(temp_dir).glob("*.jpg"))), 1)

    def test_upload_route_rejects_unsupported_extension(self):
        response = self.client.post(
            "/",
            data={
                "action": "upload",
                "image": (io.BytesIO(b"not-an-image"), "sample.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("不支持该文件格式", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
