"""
Image resizing utilities that preserve aspect ratio and important features.
"""

import cv2
import numpy as np


def resize_with_aspect_ratio(image, target_size=224, pad_value=0):
    """
    Resize image to target_size while preserving aspect ratio.
    Pads shorter dimension to make it square.

    Args:
        image: Input image (H, W, C)
        target_size: Target size for both dimensions (default 224)
        pad_value: Value to use for padding (default 0 = black)

    Returns:
        Resized and padded image of shape (target_size, target_size, C)
    """
    h, w = image.shape[:2]

    # Calculate scaling factor to fit within target_size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create square canvas
    if len(image.shape) == 3:
        canvas = np.full((target_size, target_size, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((target_size, target_size), pad_value, dtype=image.dtype)

    # Center the resized image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def resize_crop_center(image, target_size=224):
    """
    Resize image maintaining aspect ratio, then center crop.
    Better for cards that are already well-centered.

    Args:
        image: Input image (H, W, C)
        target_size: Target size for both dimensions (default 224)

    Returns:
        Center-cropped image of shape (target_size, target_size, C)
    """
    h, w = image.shape[:2]

    # Resize so the smaller dimension matches target_size
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    y_offset = (new_h - target_size) // 2
    x_offset = (new_w - target_size) // 2

    cropped = resized[y_offset:y_offset+target_size, x_offset:x_offset+target_size]

    return cropped


def adaptive_resize(image, target_size=224, strategy='pad'):
    """
    Intelligently resize based on image characteristics.

    Args:
        image: Input image (H, W, C)
        target_size: Target size (default 224)
        strategy: 'pad' (preserve all), 'crop' (focus on center), or 'stretch' (distort)

    Returns:
        Resized image of shape (target_size, target_size, C)
    """
    if strategy == 'pad':
        return resize_with_aspect_ratio(image, target_size)
    elif strategy == 'crop':
        return resize_crop_center(image, target_size)
    elif strategy == 'stretch':
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
