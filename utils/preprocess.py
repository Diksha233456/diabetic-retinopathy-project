"""
utils/preprocess.py
-------------------
Image preprocessing pipeline for Diabetic Retinopathy Detection.

Steps:
    1. Read image from disk using OpenCV
    2. Resize to 224x224 (standard for most CNN backbones)
    3. Normalize pixel values to [0, 1]
    4. Convert from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    5. Add batch dimension → final shape: (1, 3, 224, 225)
"""

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # (width, height) for cv2.resize


def load_image(image_path: str) -> np.ndarray:
    """
    Read an image from disk in BGR format using OpenCV.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        img: NumPy array of shape (H, W, 3) in BGR format, dtype uint8.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the given path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")
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
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    Args:
        img: uint8 image array.

    Returns:
        float32 image array with values in [0, 1].
    """
    return img.astype(np.float32) / 255.0


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
