import cv2, numpy as np

def lab_preprocess(img_bgr):
    """
    Convert RGB image to 6-channel representation:
    [L_CLAHE, a, b, Gx, Gy, Laplacian]

    Args:
        img_bgr: BGR image from cv2.imread (H, W, 3)

    Returns:
        6-channel float32 array normalized to [0, 1], shape (H, W, 6)
    """
    # Convert to LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # CLAHE on L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_eq = clahe.apply(L)

    # Compute derivative maps on CLAHE-enhanced L using OpenCV (10-50x faster than scikit-image)
    gx = cv2.Sobel(L_eq, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradients
    gy = cv2.Sobel(L_eq, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradients
    lap = cv2.Laplacian(L_eq, cv2.CV_64F)            # Laplacian

    # Normalize gradients to [0, 255] range
    gx = np.abs(gx)
    gy = np.abs(gy)
    lap = np.abs(lap)

    # Stack into 6-channel tensor (normalize 0–1)
    merged = np.stack([L_eq, a, b, gx, gy, lap], axis=-1)
    merged = merged.astype(np.float32) / 255.0
    return merged


def compute_edge_damage_label(img_L_channel, rim_width_pct=0.07, threshold=180):
    """
    Heuristically compute edge damage label based on rim brightness.

    Paper §6.3: "d ∈ {0,1} is a heuristic damage label estimated from
    border contrast — bright edges suggest wear/whitening."

    Args:
        img_L_channel: L channel (luminance) from LAB, uint8 (H, W)
        rim_width_pct: Outer percentage of image to consider as "rim" (default 7%)
        threshold: Mean brightness threshold for damage detection (0-255)

    Returns:
        int: 1 if edge damaged, 0 otherwise
    """
    h, w = img_L_channel.shape
    rim_h = int(h * rim_width_pct)
    rim_w = int(w * rim_width_pct)

    # Extract outer ring pixels
    top = img_L_channel[:rim_h, :]
    bottom = img_L_channel[-rim_h:, :]
    left = img_L_channel[rim_h:-rim_h, :rim_w]
    right = img_L_channel[rim_h:-rim_h, -rim_w:]

    # Compute mean brightness of rim
    rim_pixels = np.concatenate([top.flatten(), bottom.flatten(),
                                  left.flatten(), right.flatten()])
    mean_intensity = rim_pixels.mean()

    return 1 if mean_intensity > threshold else 0


def create_rim_mask(height, width, rim_width_pct=0.07):
    """
    Create binary rim mask for CBAM attention.

    Paper §5.3: "Append a simple binary rim mask B (1 on the outer
    5-8% of pixels, 0 elsewhere) to the feature tensor before attention."

    Args:
        height: Feature map height
        width: Feature map width
        rim_width_pct: Percentage of outer pixels to mark as rim (default 7%)

    Returns:
        Binary mask (H, W) with 1 on rim, 0 in center
    """
    mask = np.zeros((height, width), dtype=np.float32)
    rim_h = int(height * rim_width_pct)
    rim_w = int(width * rim_width_pct)

    # Mark outer rim as 1
    mask[:rim_h, :] = 1  # top
    mask[-rim_h:, :] = 1  # bottom
    mask[:, :rim_w] = 1  # left
    mask[:, -rim_w:] = 1  # right

    return mask
