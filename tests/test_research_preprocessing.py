import numpy as np
from PIL import Image
import pytest

from services.research.tb_vit import preprocess_reference


def test_tb_vit_constant_image_has_finite_zero_standardization():
    pytest.importorskip('cv2')
    result = preprocess_reference(Image.new('RGB', (256, 256), (70, 70, 70)))
    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, 0)


def test_tb_vit_transform_is_deterministic_and_channels_match():
    pytest.importorskip('cv2')
    pixels = np.random.default_rng(42).integers(0, 256, (192, 320, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    first = preprocess_reference(image)
    np.testing.assert_array_equal(first, preprocess_reference(image))
    np.testing.assert_array_equal(first[:, 0], first[:, 2])
    assert abs(float(first.mean())) < 1e-5
    assert abs(float(first.std())-1) < 1e-5
