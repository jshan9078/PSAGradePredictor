"""
Training Script for Dual-Branch PSA Grading Model

Implements two-phase curriculum learning:
  Phase 1: Back-only pretraining (learn defect features)
  Phase 2: Dual-branch fine-tuning (combine front + back)

Paper 8.2: "Back-only pretraining  Dual-branch fine-tuning"
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from dataset import PSADataset
from model import DualBranchPSA
from losses import CompositeLoss, compute_metrics, coral_logits_to_predictions
from sampler import create_imbalanced_sampler
from augmentations import get_train_augmentations, get_val_augmentations
from gcs_utils import save_checkpoint_to_gcs, export_model_to_gcs, download_from_gcs


def mixup_data(front, back, targets_dict, alpha=0.4):
    """
    Apply Mixup augmentation to dual-branch inputs and targets.

    Args:
        front: Front image tensor [B, C, H, W]
        back: Back image tensor [B, C, H, W]
        targets_dict: Dict with 'grade', 'edge', 'center'
        alpha: Mixup beta distribution parameter

    Returns:
        mixed_front, mixed_back, targets_a, targets_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = front.size(0)
    index = torch.randperm(batch_size, device=front.device)

    mixed_front = lam * front + (1 - lam) * front[index]
    mixed_back = lam * back + (1 - lam) * back[index]

    targets_a = targets_dict
    targets_b = {
        'grade': targets_dict['grade'][index],
        'edge': targets_dict['edge'][index],
        'center': targets_dict['center'][index]
    }

    return mixed_front, mixed_back, targets_a, targets_b, lam


def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """
    Compute mixup loss as weighted combination of two target losses.

    Args:
        criterion: CompositeLoss function
        outputs: Model outputs dict
        targets_a: First targets dict
        targets_b: Second targets dict
        lam: Mixing coefficient

    Returns:
        Mixed loss dict
    """
    loss_a = criterion(outputs, targets_a)
    loss_b = criterion(outputs, targets_b)

    # Mix all loss components
    mixed_loss = {}
    for key in loss_a.keys():
        mixed_loss[key] = lam * loss_a[key] + (1 - lam) * loss_b[key]

    return mixed_loss


def train_epoch(model, loader, criterion, optimizer, device, phase="dual", mixup_alpha=0.0, use_coral=False):
    """
    Train for one epoch.

    Args:
        model: DualBranchPSA model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        phase: "back_only" or "dual" for curriculum learning
        mixup_alpha: Mixup alpha parameter (0.0 = disabled, 0.3-0.4 recommended)
        use_coral: Whether model uses CORAL ordinal regression

    Returns:
        dict with loss and metrics
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        front = batch['front'].to(device)
        back = batch['back'].to(device)
        grade = batch['grade'].to(device)
        edge = batch['edge_damage'].to(device)
        center = batch['center'].to(device)

        optimizer.zero_grad()

        # Prepare targets
        targets_dict = {
            'grade': grade,
            'edge': edge,
            'center': center
        }

        # Apply Mixup if enabled (only in dual-branch phase)
        if mixup_alpha > 0 and phase == "dual":
            mixed_front, mixed_back, targets_a, targets_b, lam = mixup_data(
                front, back, targets_dict, alpha=mixup_alpha
            )
            outputs = model(mixed_front, mixed_back)
            loss_dict = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            # Standard forward pass
            if phase == "back_only":
                # Phase 1: Only use back branch, freeze front encoder
                # Create dummy front input (zeros)
                dummy_front = torch.zeros_like(front)
                outputs = model(dummy_front, back)
            else:
                # Phase 2: Use both branches
                outputs = model(front, back)

            loss_dict = criterion(outputs, targets_dict)

        loss = loss_dict['loss']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Collect predictions
        if use_coral:
            preds = coral_logits_to_predictions(outputs['logits'])
        else:
            preds = outputs['logits'].argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(grade.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} "
                  f"(CE: {loss_dict['loss_ce']:.4f}, EMD: {loss_dict['loss_emd']:.4f}, "
                  f"Edge: {loss_dict['loss_edge']:.4f}, Center: {loss_dict['loss_center']:.4f})")

    # Compute metrics
    metrics = compute_metrics(
        torch.tensor(all_preds),
        torch.tensor(all_targets),
        num_classes=10
    )

    return {
        'loss': total_loss / len(loader),
        'accuracy': metrics['accuracy'],
        'mae': metrics['mae'],
        'qwk': metrics['qwk']
    }


def validate(model, loader, criterion, device, phase="dual", use_coral=False):
    """Validate the model.

    Args:
        use_coral: If True, use CORAL prediction logic instead of argmax
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            front = batch['front'].to(device)
            back = batch['back'].to(device)
            grade = batch['grade'].to(device)
            edge = batch['edge_damage'].to(device)
            center = batch['center'].to(device)

            # Match training: use dummy front in back-only phase
            if phase == "back_only":
                dummy_front = torch.zeros_like(front)
                outputs = model(dummy_front, back)
            else:
                outputs = model(front, back)

            targets_dict = {
                'grade': grade,
                'edge': edge,
                'center': center
            }
            loss_dict = criterion(outputs, targets_dict)

            # Loss is already averaged per sample in the batch
            total_loss += loss_dict['loss'].item()
            num_batches += 1

            if use_coral:
                preds = coral_logits_to_predictions(outputs['logits'])
            else:
                preds = outputs['logits'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(grade.cpu().numpy())

    metrics = compute_metrics(
        torch.tensor(all_preds),
        torch.tensor(all_targets),
        num_classes=10
    )

    return {
        'loss': total_loss / num_batches,
        'accuracy': metrics['accuracy'],
        'mae': metrics['mae'],
        'qwk': metrics['qwk']
    }


def freeze_front_branch(model):
    """Freeze front encoder for back-only pretraining."""
    for param in model.front_enc.parameters():
        param.requires_grad = False
    print("Front branch frozen for back-only pretraining")


def unfreeze_front_branch(model):
    """Unfreeze front encoder for dual-branch fine-tuning."""
    for param in model.front_enc.parameters():
        param.requires_grad = True
    print("Front branch unfrozen for dual-branch training")


def save_checkpoint(checkpoint, filename, output_dir, gcs_dir=None):
    """
    Save checkpoint locally and optionally to GCS.

    Args:
        checkpoint: Checkpoint dict
        filename: Filename (e.g., 'phase2_best.pth')
        output_dir: Local output directory
        gcs_dir: Optional GCS directory (e.g., 'gs://bucket/checkpoints/')
    """
    # Save locally
    local_path = output_dir / filename
    torch.save(checkpoint, local_path)
    print(f"  → Saved checkpoint: {local_path}")

    # Upload to GCS if specified
    if gcs_dir:
        try:
            gcs_path = save_checkpoint_to_gcs(
                checkpoint=checkpoint,
                gcs_dir=gcs_dir,
                filename=filename
            )
            print(f"  → Uploaded to GCS: {gcs_path}")
        except Exception as e:
            print(f"  ⚠ Failed to upload to GCS: {e}")


def compute_class_weights(dataset, num_classes=10):
    """
    Compute class weights for imbalanced dataset.

    Args:
        dataset: PSADataset instance
        num_classes: Number of grade classes (1-10)

    Returns:
        torch.Tensor of class weights
    """
    from collections import Counter

    # Extract grade labels
    grades = []
    for item in dataset.items:
        if isinstance(item, dict):
            grades.append(item['grade'])
        else:
            grades.append(item[2])

    # Count occurrences (convert to 0-indexed)
    grade_counts = Counter([g - 1 for g in grades])

    # Compute weights: inversely proportional to frequency
    total = len(grades)
    weights = torch.zeros(num_classes)

    for class_idx in range(num_classes):
        count = grade_counts.get(class_idx, 1)  # Avoid division by zero
        weights[class_idx] = total / (num_classes * count)

    print("\nClass Weights (for CE loss):")
    for i in range(num_classes):
        print(f"  Grade {i+1}: {weights[i]:.4f} (n={grade_counts.get(i, 0)})")

    return weights


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load splits
    if args.splits_path.startswith('gs://'):
        # Download from GCS
        print(f"Downloading splits from {args.splits_path}...")
        local_splits_path = '/tmp/splits.json'
        download_from_gcs(args.splits_path, local_splits_path)
        with open(local_splits_path) as f:
            splits = json.load(f)
    else:
        with open(args.splits_path) as f:
            splits = json.load(f)

    # Create datasets
    train_aug = get_train_augmentations() if not args.no_augment else None
    val_aug = get_val_augmentations()

    train_dataset = PSADataset(
        splits['train'],
        bucket_name=args.gcs_data_bucket,
        augment=not args.no_augment,
        transform=train_aug,
        compute_edge=True,
        image_size=(args.image_size, args.image_size)
    )
    val_dataset = PSADataset(
        splits['val'],
        bucket_name=args.gcs_data_bucket,
        augment=False,
        transform=val_aug,
        compute_edge=True,
        image_size=(args.image_size, args.image_size)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset, num_classes=10).to(device)

    # Create sampler for imbalanced data
    if args.use_sampler:
        sampler = create_imbalanced_sampler(train_dataset, eta=args.sampler_eta)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    model = DualBranchPSA(
        lambda_fusion=args.lambda_fusion,
        in_channels=6,
        front_depth=args.front_depth,
        back_depth=args.back_depth,
        pretrained=args.pretrained,
        dropout=args.dropout,
        use_rim_mask=args.use_rim_mask,
        rim_mask_ratio=args.rim_mask_ratio,
        use_coral=args.use_coral
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    criterion = CompositeLoss(
        class_weights=class_weights,
        alpha_emd=args.alpha_emd,
        beta_edge=args.beta_edge,
        beta_center=args.beta_center,
        label_smoothing=args.label_smoothing,
        use_coral=args.use_coral
    )

    # ====================================
    # Phase 1: Back-Only Pretraining
    # ====================================
    if args.phase1_epochs > 0:
        print("\n" + "="*60)
        print("PHASE 1: Back-Only Pretraining")
        print("="*60)

        freeze_front_branch(model)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_phase1,
            weight_decay=args.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)

        best_val_qwk = -1
        for epoch in range(args.phase1_epochs):
            print(f"\nEpoch {epoch+1}/{args.phase1_epochs}")

            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                phase="back_only",
                mixup_alpha=0.0,  # No mixup in phase 1
                use_coral=args.use_coral
            )
            val_metrics = validate(model, val_loader, criterion, device, phase="back_only", use_coral=args.use_coral)

            scheduler.step()

            print(f"Train | Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}, "
                  f"QWK: {train_metrics['qwk']:.4f}")
            print(f"Val   | Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}, "
                  f"QWK: {val_metrics['qwk']:.4f}")

            # Save best model
            if val_metrics['qwk'] > best_val_qwk:
                best_val_qwk = val_metrics['qwk']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_qwk': best_val_qwk,
                }
                save_checkpoint(checkpoint, 'phase1_best.pth', args.output_dir, args.gcs_checkpoint_dir)
                print(f"  Best QWK: {best_val_qwk:.4f}")

    # ====================================
    # Phase 2: Dual-Branch Fine-Tuning
    # ====================================
    print("\n" + "="*60)
    print("PHASE 2: Dual-Branch Fine-Tuning")
    print("="*60)

    unfreeze_front_branch(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_phase2,
        weight_decay=args.weight_decay
    )

    # Use ReduceLROnPlateau for stable convergence
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',           # Maximize QWK
        factor=0.5,          # Reduce LR by 50% when plateau
        patience=3,          # Wait 3 epochs before reducing
        verbose=True,
        min_lr=1e-6
    )

    best_val_qwk = -1
    for epoch in range(args.phase2_epochs):
        print(f"\nEpoch {epoch+1}/{args.phase2_epochs}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            phase="dual",
            mixup_alpha=args.mixup_alpha,
            use_coral=args.use_coral
        )
        val_metrics = validate(model, val_loader, criterion, device, use_coral=args.use_coral)

        print(f"Train | Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"MAE: {train_metrics['mae']:.4f}, "
              f"QWK: {train_metrics['qwk']:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"MAE: {val_metrics['mae']:.4f}, "
              f"QWK: {val_metrics['qwk']:.4f}")

        # Step scheduler based on validation QWK
        scheduler.step(val_metrics['qwk'])

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current LR: {current_lr:.2e}")

        # Save best model
        if val_metrics['qwk'] > best_val_qwk:
            best_val_qwk = val_metrics['qwk']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_qwk': best_val_qwk,
            }
            save_checkpoint(checkpoint, 'phase2_best.pth', args.output_dir, args.gcs_checkpoint_dir)
            print(f"  Best QWK: {best_val_qwk:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, f'checkpoint_epoch_{epoch+1}.pth', args.output_dir, args.gcs_checkpoint_dir)

    print(f"\nTraining complete! Best QWK: {best_val_qwk:.4f}")

    # ====================================
    # Export Final Model
    # ====================================
    if args.gcs_model_dir:
        print("\n" + "="*60)
        print("Exporting final model to GCS")
        print("="*60)

        model_config = {
            'lambda_fusion': args.lambda_fusion,
            'in_channels': 6,
            'front_depth': 18,
            'back_depth': 34,
            'dropout': args.dropout,
            'use_rim_mask': args.use_rim_mask,
            'rim_mask_ratio': args.rim_mask_ratio,
            'alpha_emd': args.alpha_emd,
            'beta_edge': args.beta_edge,
            'beta_center': args.beta_center,
            'best_qwk': best_val_qwk,
        }

        try:
            export_model_to_gcs(
                model=model,
                model_config=model_config,
                gcs_dir=args.gcs_model_dir,
                model_name="psa_dual_branch"
            )
        except Exception as e:
            print(f"Failed to export model to GCS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PSA Grading Model")

    # Data
    parser.add_argument('--splits_path', type=str, default='splits.json')
    parser.add_argument('--output_dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (224, 384, or 448 recommended)')

    # Model
    parser.add_argument('--front_depth', type=int, default=18, choices=[18, 34, 50],
                        help='Front encoder depth (18, 34, or 50)')
    parser.add_argument('--back_depth', type=int, default=34, choices=[18, 34, 50],
                        help='Back encoder depth (18, 34, or 50)')
    parser.add_argument('--lambda_fusion', type=float, default=0.7)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--use_rim_mask', action='store_true', default=True)
    parser.add_argument('--rim_mask_ratio', type=float, default=0.07)

    # Loss weights
    parser.add_argument('--alpha_emd', type=float, default=0.7,
                        help='EMD loss coefficient (default 0.7, baseline)')
    parser.add_argument('--beta_edge', type=float, default=0.05)
    parser.add_argument('--beta_center', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for CE loss (0.0-0.2, default 0.0 for baseline)')
    parser.add_argument('--use_coral', action='store_true', default=False,
                        help='Use CORAL ordinal regression instead of CE+EMD (experimental)')

    # Training - Phase 1
    parser.add_argument('--phase1_epochs', type=int, default=10,
                        help='Back-only pretraining epochs (0 to skip)')
    parser.add_argument('--lr_phase1', type=float, default=1e-3)

    # Training - Phase 2
    parser.add_argument('--phase2_epochs', type=int, default=30)
    parser.add_argument('--lr_phase2', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Augmentation & Sampling
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--use_sampler', action='store_true', default=True)
    parser.add_argument('--sampler_eta', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Mixup alpha parameter (0.0 = disabled/baseline, 0.3-0.4 for regularization)')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=5)

    # GCS (for Vertex AI)
    parser.add_argument('--gcs_data_bucket', type=str, default=None,
                        help='GCS bucket name for image data (e.g., psa-scan-scraping-dataset)')
    parser.add_argument('--gcs_checkpoint_dir', type=str, default=None,
                        help='GCS directory for checkpoints (e.g., gs://bucket/checkpoints/)')
    parser.add_argument('--gcs_model_dir', type=str, default=None,
                        help='GCS directory for final model export (e.g., gs://bucket/models/v1/)')

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    main(args)
