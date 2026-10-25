"""
AgroDetect - Image Preprocessing Module
Preprocesses leaf images using Pillow and OpenCV for PyTorch ResNet50 inference.
Ensures identical normalization between training and inference pipelines.
"""

import io
from typing import Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
import torch
from torchvision import transforms

# ResNet50 standard specifications
TARGET_IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard PyTorch inference transformation
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def validate_image_file(file_bytes: bytes, filename: str = "") -> None:
    """
    Validates image file size, extension, and integrity.
    Raises ValueError with user-friendly message if validation fails.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty. Please select a valid leaf image.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 10 MB.")

    if filename:
        ext = filename.lower().rsplit(".", 1)
        if len(ext) < 2 or f".{ext[1]}" not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{filename}'. Allowed formats are JPG, JPEG, and PNG."
            )

    # Verify PIL can open and parse image headers
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, Exception) as exc:
        raise ValueError(
            "The uploaded file is not a valid or readable image. It may be corrupted."
        ) from exc


def load_image_with_opencv(file_bytes: bytes) -> np.ndarray:
    """
    Decodes image bytes using OpenCV into an RGB numpy array.
    Applies OpenCV color conversion from BGR to RGB.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("OpenCV failed to decode the image bytes. The image may be damaged.")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_image_with_pillow(file_bytes: bytes) -> Image.Image:
    """
    Loads image bytes with Pillow, handling color space conversions (RGBA, Grayscale, CMYK).
    Always returns an RGB PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Convert any mode (RGBA, L, CMYK, P) to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise ValueError(f"Pillow failed to process the image: {str(exc)}") from exc


def preprocess_image_opencv(file_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses leaf image using OpenCV for resizing and color conversion,
    then applies ImageNet normalization for ResNet50.
    Returns: PyTorch tensor with shape [1, 3, 224, 224]
    """
    # 1. Decode with OpenCV to RGB
    img_rgb = load_image_with_opencv(file_bytes)

    # 2. Resize using OpenCV area interpolation (high quality downsampling)
    resized_bgr = cv2.resize(img_rgb, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # 3. Convert to float tensor and scale to [0, 1]
    tensor = torch.from_numpy(resized_bgr).permute(2, 0, 1).float() / 255.0

    # 4. Normalize with ImageNet mean and standard deviation
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = normalize(tensor)

    # 5. Add batch dimension: [1, 3, 224, 224]
    return tensor.unsqueeze(0)


def preprocess_image(
    image_input: Union[bytes, Image.Image],
    filename: str = ""
) -> torch.Tensor:
    """
    Primary preprocessing function for AgroDetect.
    Accepts raw image bytes or PIL Image.
    Performs validation, RGB conversion, resizing, and ImageNet normalization.
    Returns: PyTorch tensor with shape [1, 3, 224, 224]
    """
    if isinstance(image_input, bytes):
        validate_image_file(image_input, filename=filename)
        pil_image = load_image_with_pillow(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    else:
        raise ValueError("Invalid image input type. Expected bytes or PIL Image.")

    # Apply PyTorch standard transforms
    tensor = INFERENCE_TRANSFORMS(pil_image)

    # Add batch dimension: [1, 3, 224, 224]
    return tensor.unsqueeze(0)


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization for visualization purposes.
    Accepts tensor of shape [3, 224, 224] or [1, 3, 224, 224].
    Returns: uint8 numpy array with shape [224, 224, 3] in RGB format.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    img_np = denorm.permute(1, 2, 0).cpu().numpy()
    return (img_np * 255).astype(np.uint8)
