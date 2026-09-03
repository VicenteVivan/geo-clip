from types import SimpleNamespace

import torch

from geoclip.model.image_encoder import _extract_image_features


def test_extract_image_features_supports_transformers_4_tensor():
    expected = torch.randn(2, 768)

    assert _extract_image_features(expected) is expected


def test_extract_image_features_supports_transformers_5_model_output():
    expected = torch.randn(2, 768)
    output = SimpleNamespace(pooler_output=expected)

    assert _extract_image_features(output) is expected
