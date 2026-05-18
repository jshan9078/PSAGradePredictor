"""
Ensemble Evaluation for PSA Grading Model.

Evaluates an ensemble of multiple CORAL models trained with different seeds/configurations.
Combines predictions through averaging cumulative logits for improved performance.

Expected improvement: +0.5-4.5% QWK over single model baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

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
from gcs_utils import download_from_gcs


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


def load_ensemble_model(checkpoint_path: str, device: torch.device, config: Dict) -> DualBranchPSA:
    """
    Load a single model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (local or GCS)
        device: Device to load model on
        config: Model configuration dict

    Returns:
        Loaded model in eval mode
    """
    # Download from GCS if needed
    if checkpoint_path.startswith('gs://'):
        local_path = f'/tmp/{Path(checkpoint_path).name}'
        print(f"  Downloading {checkpoint_path}...")
        download_from_gcs(checkpoint_path, local_path)
        checkpoint_path = local_path

    # Create model
    model = DualBranchPSA(
        lambda_fusion=config.get('lambda_fusion', 0.7),
        in_channels=6,
        front_depth=config.get('front_depth', 18),
        back_depth=config.get('back_depth', 34),
        pretrained=False,
        dropout=config.get('dropout', 0.25),
        use_rim_mask=config.get('use_rim_mask', True),
        rim_mask_ratio=config.get('rim_mask_ratio', 0.07),
        use_coral=True,
        num_classes=10,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_ensemble(
    models: List[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Evaluate ensemble of models.

    Args:
        models: List of trained models
        loader: Validation data loader
        device: Device to run on

    Returns:
        Dictionary with metrics (accuracy, mae, qwk, predictions, targets)
    """
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating ensemble"):
            front = batch['front'].to(device)
            back = batch['back'].to(device)
            grade = batch['grade']

            # Collect logits from all models
            batch_logits = []
            for model in models:
                outputs = model(front, back)
                logits = outputs['logits']  # [B, 9] for CORAL
                batch_logits.append(logits)

            # Average logits across ensemble
            avg_logits = torch.stack(batch_logits).mean(dim=0)  # [B, 9]

            # Convert to predictions
            preds = coral_logits_to_predictions(avg_logits)

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


def evaluate_single_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str = "model"
) -> Dict:
    """
    Evaluate a single model (for comparison).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
            front = batch['front'].to(device)
            back = batch['back'].to(device)
            grade = batch['grade']

            outputs = model(front, back)
            logits = outputs['logits']
            preds = coral_logits_to_predictions(logits)

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
    parser = argparse.ArgumentParser(description='Evaluate PSA model ensemble')

    # Data
    parser.add_argument('--splits_path', type=str, required=True,
                        help='Path to splits.json')
    parser.add_argument('--gcs_data_bucket', type=str, default=None,
                        help='GCS bucket for data (if using cloud storage)')

    # Ensemble configuration
    parser.add_argument('--ensemble_dir', type=str, default=None,
                        help='GCS directory containing ensemble models (e.g., gs://bucket/ensemble/)')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='List of checkpoint paths (alternative to --ensemble_dir)')

    # Model configuration (default values, can be overridden per model)
    parser.add_argument('--lambda_fusion', type=float, default=0.7)
    parser.add_argument('--front_depth', type=int, default=18)
    parser.add_argument('--back_depth', type=int, default=34)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--use_rim_mask', action='store_true', default=True)
    parser.add_argument('--rim_mask_ratio', type=float, default=0.07)

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_individual', action='store_true',
                        help='Also evaluate each model individually')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Determine checkpoint paths
    if args.ensemble_dir:
        # Auto-discover checkpoints in ensemble directory
        # Expected structure: ensemble/model1/checkpoints/phase2_best.pth
        #                     ensemble/model2/checkpoints/phase2_best.pth
        #                     ...
        from google.cloud import storage
        client = storage.Client()

        # Parse GCS path
        if not args.ensemble_dir.startswith('gs://'):
            raise ValueError("ensemble_dir must start with gs://")

        path_parts = args.ensemble_dir[5:].split('/', 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''

        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Find all phase2_best.pth files
        checkpoint_paths = []
        for blob in blobs:
            if blob.name.endswith('phase2_best.pth'):
                checkpoint_paths.append(f'gs://{bucket_name}/{blob.name}')

        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {args.ensemble_dir}")

        print(f"Found {len(checkpoint_paths)} checkpoints in ensemble directory:")
        for cp in checkpoint_paths:
            print(f"  - {cp}")
        print()

    elif args.checkpoints:
        checkpoint_paths = args.checkpoints
        print(f"Using {len(checkpoint_paths)} manually specified checkpoints:")
        for cp in checkpoint_paths:
            print(f"  - {cp}")
        print()

    else:
        raise ValueError("Must specify either --ensemble_dir or --checkpoints")

    # Load validation dataset
    print(f"Loading validation data from {args.splits_path}...")
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
    print()

    # Load ensemble models
    print(f"Loading {len(checkpoint_paths)} ensemble models...")
    models = []
    model_configs = []

    for i, cp in enumerate(checkpoint_paths):
        # Parse model name from path
        # Expected: gs://bucket/ensemble/model1/checkpoints/phase2_best.pth
        model_name = Path(cp).parent.parent.name

        # Default config (can be customized per model in future)
        config = {
            'lambda_fusion': args.lambda_fusion,
            'front_depth': args.front_depth,
            'back_depth': args.back_depth,
            'dropout': args.dropout,
            'use_rim_mask': args.use_rim_mask,
            'rim_mask_ratio': args.rim_mask_ratio,
        }

        # Load model-specific config if it exists
        config_path = str(Path(cp).parent / 'config.json')
        if config_path.startswith('/tmp'):
            # Try GCS path
            gcs_config_path = cp.replace('phase2_best.pth', 'config.json')
            try:
                local_config = '/tmp/model_config.json'
                download_from_gcs(gcs_config_path, local_config)
                with open(local_config, 'r') as f:
                    model_config = json.load(f)
                config.update(model_config)
            except:
                pass  # Use default config

        print(f"  [{i+1}/{len(checkpoint_paths)}] Loading {model_name}...")
        model = load_ensemble_model(cp, device, config)
        models.append(model)
        model_configs.append(config)

    print(f"✓ Loaded {len(models)} models")
    print()

    # Evaluate individual models (optional)
    if args.eval_individual:
        print("="*60)
        print("INDIVIDUAL MODEL EVALUATION")
        print("="*60)
        print()

        individual_results = []
        for i, (model, cp) in enumerate(zip(models, checkpoint_paths)):
            model_name = Path(cp).parent.parent.name
            result = evaluate_single_model(model, val_loader, device, model_name)
            individual_results.append(result)

            print(f"{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  MAE:      {result['mae']:.4f}")
            print(f"  QWK:      {result['qwk']:.4f}")
            print()

        # Show statistics
        qwks = [r['qwk'] for r in individual_results]
        print(f"Individual model QWK statistics:")
        print(f"  Mean:   {np.mean(qwks):.4f}")
        print(f"  Std:    {np.std(qwks):.4f}")
        print(f"  Min:    {np.min(qwks):.4f}")
        print(f"  Max:    {np.max(qwks):.4f}")
        print()

    # Evaluate ensemble
    print("="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    print()

    ensemble_results = evaluate_ensemble(models, val_loader, device)

    print(f"Ensemble Results ({len(models)} models):")
    print(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"  MAE:      {ensemble_results['mae']:.4f}")
    print(f"  QWK:      {ensemble_results['qwk']:.4f}")
    print()

    # Compare to individual models if evaluated
    if args.eval_individual:
        print("="*60)
        print("IMPROVEMENT ANALYSIS")
        print("="*60)
        print()

        best_individual_qwk = np.max([r['qwk'] for r in individual_results])
        mean_individual_qwk = np.mean([r['qwk'] for r in individual_results])

        improvement_vs_best = ensemble_results['qwk'] - best_individual_qwk
        improvement_vs_mean = ensemble_results['qwk'] - mean_individual_qwk

        print(f"Ensemble QWK:          {ensemble_results['qwk']:.4f}")
        print(f"Best individual QWK:   {best_individual_qwk:.4f}")
        print(f"Mean individual QWK:   {mean_individual_qwk:.4f}")
        print()
        print(f"Improvement vs best:   {improvement_vs_best:+.4f} ({improvement_vs_best/best_individual_qwk*100:+.2f}%)")
        print(f"Improvement vs mean:   {improvement_vs_mean:+.4f} ({improvement_vs_mean/mean_individual_qwk*100:+.2f}%)")
        print()

        if improvement_vs_best > 0:
            print(f"✅ Ensemble improves over best individual model by {improvement_vs_best:.4f} QWK!")
        else:
            print(f"⚠️  Ensemble does not improve over best individual model (delta: {improvement_vs_best:.4f})")
            print("    This might indicate:")
            print("    - Models are too similar (not enough diversity)")
            print("    - Individual models already very strong")
            print("    - Need more ensemble members")


if __name__ == '__main__':
    main()
