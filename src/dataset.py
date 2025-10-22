import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import io
from google.cloud import storage
from preprocess import lab_preprocess, compute_edge_damage_label
from resize_utils import resize_with_aspect_ratio

class PSADataset(Dataset):
    """
    PSA Card Grading Dataset

    Args:
        manifest: List of dicts with keys: 'front', 'back', 'grade', 'cert_id'
        bucket_name: GCS bucket name (if using cloud storage)
        augment: Whether to apply augmentations
        transform: Augmentation transform (e.g., from albumentations)
        compute_edge: Whether to compute edge damage labels
    """
    def __init__(self, manifest, bucket_name=None, augment=False, transform=None, compute_edge=True, image_size=(224, 224)):
        self.items = manifest
        self.bucket_name = bucket_name
        self.augment = augment
        self.transform = transform
        self.compute_edge = compute_edge
        self.image_size = image_size  # (height, width)

        # Don't initialize GCS client here - create it lazily per-worker
        # to avoid pickling issues with multiprocessing
        self._gcs_client = None
        self._bucket = None

    def __len__(self):
        return len(self.items)

    @property
    def bucket(self):
        """Lazy initialization of GCS bucket (per-worker for multiprocessing)."""
        if self.bucket_name and self._bucket is None:
            self._gcs_client = storage.Client()
            self._bucket = self._gcs_client.bucket(self.bucket_name)
        return self._bucket

    def _load_image_from_gcs(self, blob_path: str) -> np.ndarray:
        """
        Load image from GCS bucket.

        Args:
            blob_path: Path to blob in GCS (e.g., 'png/8/12345_front.png')

        Returns:
            Image as numpy array in BGR format (OpenCV convention)
        """
        blob = self.bucket.blob(blob_path)
        image_bytes = blob.download_as_bytes()

        # Decode image bytes to numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image from GCS: {blob_path}")

        return image

    def __getitem__(self, idx):
        item = self.items[idx]

        # Handle both dict and tuple formats for backwards compatibility
        if isinstance(item, dict):
            front_path = item['front']
            back_path = item['back']
            grade = item['grade']
        else:
            front_path, back_path, grade = item

        # Load images (BGR from cv2)
        if self.bucket_name:
            # Load from GCS
            front_bgr = self._load_image_from_gcs(front_path)
            back_bgr = self._load_image_from_gcs(back_path)
        else:
            # Load from local filesystem
            front_bgr = cv2.imread(front_path)
            back_bgr = cv2.imread(back_path)

        # Convert BGR to RGB for preprocessing
        front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
        back_rgb = cv2.cvtColor(back_bgr, cv2.COLOR_BGR2RGB)

        # Compute edge damage label from back image (before CLAHE)
        if self.compute_edge:
            back_lab = cv2.cvtColor(back_bgr, cv2.COLOR_BGR2LAB)
            L_channel = back_lab[:, :, 0]
            edge_label = compute_edge_damage_label(L_channel)
        else:
            edge_label = 0

        # Apply LAB + CLAHE + gradient preprocessing
        front_6ch = lab_preprocess(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
        back_6ch = lab_preprocess(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR))

        # Resize to fixed size while preserving aspect ratio
        # This pads the image rather than distorting it
        if front_6ch.shape[:2] != self.image_size:
            front_6ch = resize_with_aspect_ratio(front_6ch, target_size=self.image_size[0], pad_value=0)
        if back_6ch.shape[:2] != self.image_size:
            back_6ch = resize_with_aspect_ratio(back_6ch, target_size=self.image_size[0], pad_value=0)

        # Apply augmentations if enabled
        if self.augment and self.transform:
            front_6ch = self.transform(image=front_6ch)["image"]
            back_6ch = self.transform(image=back_6ch)["image"]

        # Convert to tensors (H, W, C) -> (C, H, W)
        front_tensor = torch.from_numpy(front_6ch).permute(2, 0, 1).float()
        back_tensor = torch.from_numpy(back_6ch).permute(2, 0, 1).float()

        # Grade label (convert to 0-indexed)
        grade_label = torch.tensor(grade - 1, dtype=torch.long)

        # Edge damage label
        edge_label = torch.tensor(edge_label, dtype=torch.float32)

        # Centering target: for now, assume perfect centering (0.5, 0.5)
        # In a real implementation, you'd extract this from annotations or estimate it
        center_target = torch.tensor([0.5, 0.5], dtype=torch.float32)

        return {
            'front': front_tensor,
            'back': back_tensor,
            'grade': grade_label,
            'edge_damage': edge_label,
            'center': center_target,
        }
