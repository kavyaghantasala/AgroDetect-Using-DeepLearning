"""
AgroDetect - Unit Tests for Preprocessing Module
Tests image validation, format conversions, dimensions, and ImageNet normalization.
"""

import io
import pytest
import numpy as np
from PIL import Image
import torch

from app.preprocessing import (
    validate_image_file,
    load_image_with_pillow,
    load_image_with_opencv,
    preprocess_image,
    preprocess_image_opencv,
    denormalize_tensor,
    TARGET_IMAGE_SIZE,
    MAX_FILE_SIZE_BYTES
)


def create_test_image_bytes(format="JPEG", size=(300, 300), color=(100, 180, 50), mode="RGB") -> bytes:
    """Helper to generate in-memory synthetic test image bytes."""
    img = Image.new(mode, size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


class TestImagePreprocessing:

    def test_valid_jpeg_image_preprocessing(self):
        """Tests that a standard valid JPEG is processed into a (1, 3, 224, 224) float tensor."""
        img_bytes = create_test_image_bytes(format="JPEG", size=(500, 400))
        tensor = preprocess_image(img_bytes, filename="leaf.jpg")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_valid_png_image_preprocessing(self):
        """Tests that PNG images are supported and correctly processed."""
        img_bytes = create_test_image_bytes(format="PNG", size=(250, 250))
        tensor = preprocess_image(img_bytes, filename="leaf.png")

        assert tensor.shape == (1, 3, 224, 224)

    def test_rgba_image_converted_to_3_channels(self):
        """Tests that RGBA 4-channel images are safely converted to RGB 3-channel tensors."""
        img_bytes = create_test_image_bytes(format="PNG", size=(200, 200), mode="RGBA", color=(50, 150, 50, 200))
        tensor = preprocess_image(img_bytes, filename="leaf_transparent.png")

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.shape[1] == 3

    def test_grayscale_image_converted_to_3_channels(self):
        """Tests that 1-channel Grayscale images are converted to 3 RGB channels."""
        img = Image.new("L", (200, 200), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        tensor = preprocess_image(buffer.getvalue(), filename="gray_leaf.jpg")

        assert tensor.shape == (1, 3, 224, 224)

    def test_opencv_preprocessing_pipeline(self):
        """Tests OpenCV decode and preprocessing pathway."""
        img_bytes = create_test_image_bytes(format="JPEG", size=(400, 300))
        tensor = preprocess_image_opencv(img_bytes)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_corrupted_image_raises_value_error(self):
        """Tests that corrupted non-image bytes raise a clear ValueError."""
        corrupted_bytes = b"This is not a real image file, just corrupted random text."

        with pytest.raises(ValueError) as excinfo:
            validate_image_file(corrupted_bytes, filename="corrupted.jpg")

        assert "not a valid or readable image" in str(excinfo.value)

    def test_empty_image_raises_value_error(self):
        """Tests that empty 0-byte input raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            validate_image_file(b"", filename="empty.jpg")

        assert "empty" in str(excinfo.value)

    def test_unsupported_extension_raises_value_error(self):
        """Tests that disallowed extensions (.pdf, .exe) raise ValueError."""
        img_bytes = create_test_image_bytes(format="JPEG")
        with pytest.raises(ValueError) as excinfo:
            validate_image_file(img_bytes, filename="leaf.pdf")

        assert "Unsupported file format" in str(excinfo.value)

    def test_file_size_exceeding_limit_raises_value_error(self):
        """Tests that oversized image files (>10MB) are rejected."""
        fake_large_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
        with pytest.raises(ValueError) as excinfo:
            validate_image_file(fake_large_bytes, filename="huge.jpg")

        assert "exceeds maximum allowed limit" in str(excinfo.value)

    def test_denormalization_helper(self):
        """Tests that denormalization transforms normalized tensor back to uint8 [0, 255] image."""
        img_bytes = create_test_image_bytes(format="JPEG", size=(224, 224))
        tensor = preprocess_image(img_bytes)
        denorm_img = denormalize_tensor(tensor)

        assert isinstance(denorm_img, np.ndarray)
        assert denorm_img.shape == (224, 224, 3)
        assert denorm_img.dtype == np.uint8
