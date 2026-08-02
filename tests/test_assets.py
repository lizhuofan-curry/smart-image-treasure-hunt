import os
import unittest

import numpy as np
import torch

from image_classification.classification_config import CLASSIFIER_MODEL_PATH
from image_classification.classification_model import Classifier
from image_denoising.denoising_config import DENOISER_MODEL_PATH
from image_denoising.denoising_model import ConvDenoiser
from image_similarity.similarity_config import (
    DECODER_MODEL_PATH,
    EMBEDDING_PATH,
    ENCODER_MODEL_PATH,
)
from image_similarity.similarity_model import ConvDecoder, ConvEncoder


@unittest.skipUnless(
    os.getenv("RUN_LFS_ASSET_TESTS") == "1",
    "仅在手动 CI 下载 Git LFS 资产后运行",
)
class LfsAssetCompatibilityTests(unittest.TestCase):
    def test_weights_match_current_model_definitions(self):
        assets = (
            (Classifier(), CLASSIFIER_MODEL_PATH),
            (ConvDenoiser(), DENOISER_MODEL_PATH),
            (ConvEncoder(), ENCODER_MODEL_PATH),
            (ConvDecoder(), DECODER_MODEL_PATH),
        )

        for model, weight_path in assets:
            with self.subTest(weight_path=str(weight_path)):
                state_dict = torch.load(
                    weight_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state_dict)

    def test_embedding_library_is_1024_dimensions(self):
        embeddings = np.load(EMBEDDING_PATH, mmap_mode="r")

        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.shape[1], 1024)
        self.assertGreater(embeddings.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
