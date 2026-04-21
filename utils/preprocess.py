"""
utils/preprocess.py
-------------------
Image preprocessing pipeline for Diabetic Retinopathy Detection.

Steps:
    1. Read image from disk using OpenCV (BGR → RGB)
    2. Resize to 224×224 (standard for ResNet-50)
    3. Normalize pixel values to [0, 1]
    4. Apply ImageNet channel-wise mean/std normalization
    5. Convert from HWC to CHW and add batch dimension → (1, 3, 224, 224)
"""

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # (width, height) for cv2.resize

# ImageNet normalization constants (required for ResNet-50 pretrained features)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(image_path: str) -> np.ndarray:
    """
    Read an image from disk. Converts from OpenCV BGR to RGB.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        img: NumPy array of shape (H, W, 3) in RGB format, dtype uint8.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the given path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")
    # OpenCV reads as BGR — convert to RGB for ImageNet-style normalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Resize the image to the target size.

    Args:
        img:  Input image array (H, W, 3).
        size: Target (width, height). Default is (224, 224).

    Returns:
        Resized image array of shape (size[1], size[0], 3).
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] then apply ImageNet mean/std.

    Args:
        img: uint8 RGB image array of shape (H, W, 3).

    Returns:
        float32 array normalized with ImageNet statistics.
    """
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def to_tensor_format(img: np.ndarray) -> np.ndarray:
    """
    Convert image from HWC (OpenCV default) to CHW (PyTorch default),
    then add a batch dimension at axis 0.

    Args:
        img: float32 array of shape (H, W, 3).

    Returns:
        float32 array of shape (1, 3, H, W).
    """
    # HWC → CHW
    img_chw = np.transpose(img, (2, 0, 1))
    # CHW → NCHW  (add batch dimension)
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch


def preprocess(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → resize → normalize → tensor format.

    Applies ImageNet mean/std normalization as required by ResNet-50.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed NumPy array of shape (1, 3, 224, 224), dtype float32.

    Example:
        >>> tensor = preprocess("data/sample_images/eye.jpg")
        >>> print(tensor.shape)   # (1, 3, 224, 224)
    """
    img = load_image(image_path)
    img = resize_image(img)
    img = normalize_image(img)
    img = to_tensor_format(img)
    return img
