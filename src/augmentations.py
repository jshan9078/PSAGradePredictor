"""
Data augmentations for PSA card grading.

Uses Albumentations library for image augmentation.
Augmentations are applied AFTER LAB preprocessing (on 6-channel tensors).
"""

import albumentations as A


def get_train_augmentations():
    """
    Training augmentations for both front and back images.

    Applied to 6-channel tensors (LAB + gradients) after preprocessing.
    """
    return A.Compose([
        # Geometric augmentations
        A.Rotate(limit=2, border_mode=0, p=0.5),
        A.Affine(translate_percent=0.02, scale=(0.98, 1.02), p=0.5),
        A.Perspective(scale=(0.0, 0.02), p=0.3),

        # Brightness/Contrast (affects L channel primarily)
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),

        # Noise (simulate scan artifacts) - try both parameter names for compatibility
        A.GaussNoise(variance_limit=(5.0, 15.0), p=0.3),

        # Blur (simulate focus issues)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_val_augmentations():
    """
    Validation augmentations (minimal/none).

    Returns None or identity transform for consistency.
    """
    return None  # No augmentation for validation


# Legacy function for backwards compatibility
def get_transforms(branch=None):
    """
    Legacy function - returns training augmentations.
    Use get_train_augmentations() instead.

    Args:
        branch: Ignored (kept for backwards compatibility)
    """
    return get_train_augmentations()
