"""
Test-Time Augmentation (TTA) evaluation for PSA grading model.

This script evaluates a trained model with TTA to boost performance without retraining.
TTA applies multiple augmentations to each test sample and averages predictions.

Expected improvement: +0.01-0.02 QWK over single-crop inference.
"""

import argparse
import os
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import PSADataset
from model import DualBranchPSA


def coral_logits_to_predictions(cumulative_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL cumulative logits to integer predictions.

    Args:
        cumulative_logits: [B, num_classes-1] cumulative binary logits

    Returns:
        [B] integer predictions in [0, num_classes-1]
    """
    # Get cumulative probabilities P(y > k)
    cum_probs = torch.sigmoid(cumulative_logits)  # [B, 9]

    # Pad boundaries: P(y > -1) = 1, P(y > 9) = 0
    cum_probs = torch.cat([
        torch.ones(cum_probs.size(0), 1, device=cum_probs.device),
        cum_probs,
        torch.zeros(cum_probs.size(0), 1, device=cum_probs.device)
    ], dim=1)  # [B, 11]

    # Class probabilities: P(y = k) = P(y > k-1) - P(y > k)
    probs = cum_probs[:, :-1] - cum_probs[:, 1:]  # [B, 10]

    # Predict class with highest probability
    preds = torch.argmax(probs, dim=1)  # [B]

    return preds


def get_tta_transforms():
    """
    Define test-time augmentations.

    Keep augmentations light and reversible:
    - Small rotations (cards can be slightly tilted in scans)
    - Brightness variations (lighting/scan conditions)
    - Horizontal flip (symmetry)

    Avoid heavy augmentations that might degrade card features.
    """
    transforms = [
        # Identity (no augmentation)
        None,

        # Horizontal flip (symmetry)
        A.HorizontalFlip(p=1.0),

        # Small rotations (±3 degrees)
        A.Rotate(limit=(-3, -3), border_mode=0, p=1.0),
        A.Rotate(limit=(3, 3), border_mode=0, p=1.0),

        # Brightness variations (scan conditions)
        A.RandomBrightnessContrast(brightness_limit=(0.05, 0.05), contrast_limit=0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.05, -0.05), contrast_limit=0, p=1.0),
    ]

    return transforms


def evaluate_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_transforms: list,
    use_coral: bool = True,
) -> dict:
    """
    Evaluate model with test-time augmentation.

    Args:
        model: Trained PSA grading model
        loader: Validation data loader
        device: Device to run on
        tta_transforms: List of augmentation transforms (None for identity)
        use_coral: Use CORAL prediction logic

    Returns:
        Dictionary with metrics (accuracy, mae, qwk)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating with TTA"):
            front = batch['front'].cpu().numpy()  # [B, 6, H, W]
            back = batch['back'].cpu().numpy()
            grade = batch['grade']

            batch_size = front.shape[0]
            batch_logits = []  # Collect logits from all TTA variants

            # Apply each TTA transform
            for transform in tta_transforms:
                front_aug = front.copy()
                back_aug = back.copy()

                if transform is not None:
                    # Apply augmentation to each sample in batch
                    for i in range(batch_size):
                        # Albumentations expects (H, W, C), we have (C, H, W)
                        front_i = front[i].transpose(1, 2, 0)  # [H, W, 6]
                        back_i = back[i].transpose(1, 2, 0)

                        # Apply same augmentation to both front and back
                        augmented_front = transform(image=front_i)['image']
                        augmented_back = transform(image=back_i)['image']

                        front_aug[i] = augmented_front.transpose(2, 0, 1)  # [6, H, W]
                        back_aug[i] = augmented_back.transpose(2, 0, 1)

                # Convert to tensor and move to device
                front_tensor = torch.from_numpy(front_aug).to(device)
                back_tensor = torch.from_numpy(back_aug).to(device)

                # Forward pass
                outputs = model(front_tensor, back_tensor)
                logits = outputs['logits']  # [B, 9] for CORAL or [B, 10] for CE

                batch_logits.append(logits)

            # Average logits across all TTA variants
            avg_logits = torch.stack(batch_logits).mean(dim=0)  # [B, 9 or 10]

            # Convert to predictions
            if use_coral:
                preds = coral_logits_to_predictions(avg_logits)
            else:
                preds = avg_logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(grade.numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')

    return {
        'accuracy': accuracy,
        'mae': mae,
        'qwk': qwk,
        'predictions': all_preds,
        'targets': all_targets
    }


def evaluate_baseline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_coral: bool = True,
) -> dict:
    """
    Evaluate model without TTA (baseline single-crop inference).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating baseline"):
            front = batch['front'].to(device)
            back = batch['back'].to(device)
            grade = batch['grade']

            outputs = model(front, back)
            logits = outputs['logits']

            if use_coral:
                preds = coral_logits_to_predictions(logits)
            else:
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(grade.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')

    return {
        'accuracy': accuracy,
        'mae': mae,
        'qwk': qwk,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSA model with TTA')

    # Data
    parser.add_argument('--splits_path', type=str, required=True,
                        help='Path to splits.json')
    parser.add_argument('--gcs_data_bucket', type=str, default=None,
                        help='GCS bucket for data (if using cloud storage)')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--lambda_fusion', type=float, default=0.7)
    parser.add_argument('--front_depth', type=int, default=18)
    parser.add_argument('--back_depth', type=int, default=34)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--use_rim_mask', action='store_true', default=True)
    parser.add_argument('--rim_mask_ratio', type=float, default=0.07)
    parser.add_argument('--use_coral', action='store_true', default=False)

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation (only run TTA)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = DualBranchPSA(
        lambda_fusion=args.lambda_fusion,
        in_channels=6,
        front_depth=args.front_depth,
        back_depth=args.back_depth,
        pretrained=False,  # Loading from checkpoint
        dropout=args.dropout,
        use_rim_mask=args.use_rim_mask,
        rim_mask_ratio=args.rim_mask_ratio,
        use_coral=args.use_coral,
        num_classes=10,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Handle backward compatibility: filter out attention fusion keys if present
    state_dict = checkpoint['model_state_dict']
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Remove keys that don't match (attention fusion keys)
    keys_to_remove = checkpoint_keys - model_keys
    if keys_to_remove:
        print(f"Warning: Checkpoint contains {len(keys_to_remove)} extra keys (likely from attention fusion)")
        print("Filtering them out for backward compatibility...")
        for key in keys_to_remove:
            del state_dict[key]

    # Check if we're missing keys
    missing_keys = model_keys - checkpoint_keys
    if missing_keys:
        print(f"Error: Model expects {len(missing_keys)} keys not in checkpoint:")
        for key in list(missing_keys)[:5]:
            print(f"  - {key}")
        raise RuntimeError("Cannot load checkpoint: model architecture mismatch")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load validation dataset
    print(f"\nLoading validation data from {args.splits_path}...")

    import json
    with open(args.splits_path, 'r') as f:
        splits = json.load(f)

    val_dataset = PSADataset(
        manifest=splits['val'],
        bucket_name=args.gcs_data_bucket,
        augment=False,
        transform=None,
        compute_edge=True,
        image_size=(args.image_size, args.image_size)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"✓ Loaded {len(val_dataset)} validation samples")

    # Baseline evaluation (no TTA)
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("BASELINE EVALUATION (Single-Crop Inference)")
        print("="*60)
        baseline_results = evaluate_baseline(model, val_loader, device, args.use_coral)

        print(f"\nBaseline Results:")
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  MAE:      {baseline_results['mae']:.4f}")
        print(f"  QWK:      {baseline_results['qwk']:.4f}")

    # TTA evaluation
    print("\n" + "="*60)
    print("TTA EVALUATION (Multi-Crop Averaging)")
    print("="*60)

    tta_transforms = get_tta_transforms()
    print(f"Using {len(tta_transforms)} TTA variants:")
    print("  1. Identity (no augmentation)")
    print("  2. Horizontal flip")
    print("  3. Rotate -3°")
    print("  4. Rotate +3°")
    print("  5. Brightness +5%")
    print("  6. Brightness -5%")
    print()

    tta_results = evaluate_with_tta(model, val_loader, device, tta_transforms, args.use_coral)

    print(f"\nTTA Results:")
    print(f"  Accuracy: {tta_results['accuracy']:.4f}")
    print(f"  MAE:      {tta_results['mae']:.4f}")
    print(f"  QWK:      {tta_results['qwk']:.4f}")

    # Compare baseline vs TTA
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("IMPROVEMENT ANALYSIS")
        print("="*60)

        acc_delta = tta_results['accuracy'] - baseline_results['accuracy']
        mae_delta = tta_results['mae'] - baseline_results['mae']
        qwk_delta = tta_results['qwk'] - baseline_results['qwk']

        print(f"\nTTA vs Baseline:")
        print(f"  Accuracy: {acc_delta:+.4f} ({acc_delta/baseline_results['accuracy']*100:+.2f}%)")
        print(f"  MAE:      {mae_delta:+.4f} ({mae_delta/baseline_results['mae']*100:+.2f}%)")
        print(f"  QWK:      {qwk_delta:+.4f} ({qwk_delta/baseline_results['qwk']*100:+.2f}%)")

        if qwk_delta > 0:
            print(f"\n✅ TTA improves QWK by {qwk_delta:.4f} ({qwk_delta*100:.2f} percentage points)")
            print(f"   This is a {qwk_delta/baseline_results['qwk']*100:.1f}% relative improvement!")
        else:
            print(f"\n❌ TTA does not improve QWK (delta: {qwk_delta:.4f})")
            print(f"   This might indicate model is already robust to these augmentations.")


if __name__ == '__main__':
    main()
