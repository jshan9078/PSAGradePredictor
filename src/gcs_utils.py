"""
Google Cloud Storage utilities for Vertex AI training.

Handles checkpoint saving/loading and model export to GCS.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from google.cloud import storage


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """
    Parse GCS path into bucket and blob name.

    Args:
        gcs_path: Path like 'gs://bucket-name/path/to/file'

    Returns:
        (bucket_name, blob_name) tuple
    """
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"GCS path must start with gs://, got: {gcs_path}")

    path = gcs_path[5:]  # Remove 'gs://'
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ''

    return bucket_name, blob_name


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Upload a local file to GCS.

    Args:
        local_path: Path to local file
        gcs_path: Destination GCS path (gs://bucket/path/to/file)
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → {gcs_path}")


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """
    Download a file from GCS to local storage.

    Args:
        gcs_path: Source GCS path (gs://bucket/path/to/file)
        local_path: Destination local path
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create parent directory if needed
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} → {local_path}")


def save_checkpoint_to_gcs(
    checkpoint: Dict[str, Any],
    gcs_dir: str,
    filename: str,
    local_dir: str = "/tmp/checkpoints"
) -> str:
    """
    Save a PyTorch checkpoint to GCS.

    Args:
        checkpoint: Checkpoint dict with model state, optimizer state, etc.
        gcs_dir: GCS directory (gs://bucket/path/to/checkpoints/)
        filename: Checkpoint filename (e.g., 'phase2_best.pth')
        local_dir: Temporary local directory for saving

    Returns:
        Full GCS path where checkpoint was saved
    """
    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Save checkpoint locally first
    local_path = os.path.join(local_dir, filename)
    torch.save(checkpoint, local_path)

    # Upload to GCS
    gcs_path = f"{gcs_dir.rstrip('/')}/{filename}"
    upload_to_gcs(local_path, gcs_path)

    # Clean up local file
    os.remove(local_path)

    return gcs_path


def load_checkpoint_from_gcs(
    gcs_path: str,
    local_dir: str = "/tmp/checkpoints",
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint from GCS.

    Args:
        gcs_path: Full GCS path to checkpoint
        local_dir: Temporary local directory
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')

    Returns:
        Checkpoint dictionary
    """
    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Extract filename from GCS path
    filename = gcs_path.split('/')[-1]
    local_path = os.path.join(local_dir, filename)

    # Download from GCS
    download_from_gcs(gcs_path, local_path)

    # Load checkpoint
    checkpoint = torch.load(local_path, map_location=map_location)

    # Clean up local file
    os.remove(local_path)

    return checkpoint


def export_model_to_gcs(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    gcs_dir: str,
    model_name: str = "model"
) -> Dict[str, str]:
    """
    Export model state dict and config to GCS for deployment.

    Args:
        model: Trained PyTorch model
        model_config: Model configuration dict (hyperparameters, architecture, etc.)
        gcs_dir: GCS directory for model export (gs://bucket/models/v1/)
        model_name: Base name for model files

    Returns:
        Dict with GCS paths for state_dict and config
    """
    local_dir = "/tmp/export"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Save state dict
    state_dict_filename = f"{model_name}_state_dict.pth"
    state_dict_local = os.path.join(local_dir, state_dict_filename)
    torch.save(model.state_dict(), state_dict_local)

    # Save config
    config_filename = f"{model_name}_config.json"
    config_local = os.path.join(local_dir, config_filename)
    with open(config_local, 'w') as f:
        json.dump(model_config, f, indent=2)

    # Upload both files
    state_dict_gcs = f"{gcs_dir.rstrip('/')}/{state_dict_filename}"
    config_gcs = f"{gcs_dir.rstrip('/')}/{config_filename}"

    upload_to_gcs(state_dict_local, state_dict_gcs)
    upload_to_gcs(config_local, config_gcs)

    # Clean up
    os.remove(state_dict_local)
    os.remove(config_local)

    print(f"\nModel exported to GCS:")
    print(f"  State dict: {state_dict_gcs}")
    print(f"  Config:     {config_gcs}")

    return {
        'state_dict': state_dict_gcs,
        'config': config_gcs
    }


def sync_splits_to_gcs(splits_path: str, gcs_bucket: str) -> str:
    """
    Upload splits.json to GCS for training access.

    Args:
        splits_path: Local path to splits.json
        gcs_bucket: GCS bucket name (without gs://)

    Returns:
        GCS path where splits were uploaded
    """
    gcs_path = f"gs://{gcs_bucket}/data/splits.json"
    upload_to_gcs(splits_path, gcs_path)
    return gcs_path
