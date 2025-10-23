#!/usr/bin/env python3
"""
Quick test to verify CORAL implementation works.
Run this before deploying to ensure no errors.
"""

import torch
import sys
sys.path.insert(0, 'src')

from model import DualBranchPSA
from losses import CompositeLoss, coral_logits_to_predictions

def test_coral_model():
    """Test CORAL model forward pass."""
    print("Testing CORAL model...")

    # Create CORAL model
    model = DualBranchPSA(
        use_coral=True,
        num_classes=10,
        front_depth=18,
        back_depth=18,  # Smaller for faster test
        pretrained=False
    )

    # Dummy inputs
    batch_size = 4
    front = torch.randn(batch_size, 6, 224, 224)
    back = torch.randn(batch_size, 6, 224, 224)

    # Forward pass
    outputs = model(front, back)

    # Check output shapes
    assert outputs['logits'].shape == (batch_size, 9), f"Expected [4, 9], got {outputs['logits'].shape}"
    assert outputs['edge_logit'].shape == (batch_size,), f"Expected [4], got {outputs['edge_logit'].shape}"
    assert outputs['center'].shape == (batch_size, 2), f"Expected [4, 2], got {outputs['center'].shape}"

    print("✅ CORAL model forward pass: PASSED")
    return outputs

def test_coral_loss():
    """Test CORAL loss computation."""
    print("\nTesting CORAL loss...")

    # Create CORAL loss
    criterion = CompositeLoss(use_coral=True, num_classes=10)

    # Dummy model outputs
    batch_size = 4
    outputs = {
        'logits': torch.randn(batch_size, 9),  # 9 cumulative logits
        'edge_logit': torch.randn(batch_size),
        'center': torch.randn(batch_size, 2)
    }

    # Dummy targets
    targets = {
        'grade': torch.randint(0, 10, (batch_size,)),
        'edge': torch.randint(0, 2, (batch_size,)).float(),
        'center': torch.randn(batch_size, 2)
    }

    # Compute loss
    loss_dict = criterion(outputs, targets)

    # Check loss values
    assert 'loss' in loss_dict, "Missing 'loss' in output"
    assert loss_dict['loss'].item() > 0, "Loss should be positive"
    assert not torch.isnan(loss_dict['loss']), "Loss is NaN!"

    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    print("✅ CORAL loss computation: PASSED")
    return loss_dict

def test_coral_predictions():
    """Test CORAL prediction conversion."""
    print("\nTesting CORAL predictions...")

    # Test case: clear predictions
    # If all logits are positive, grade should be 9 (all thresholds exceeded)
    cum_logits = torch.ones(1, 9) * 5.0
    preds = coral_logits_to_predictions(cum_logits)
    assert preds[0] == 9, f"Expected 9, got {preds[0]}"

    # If all logits are negative, grade should be 0 (no thresholds exceeded)
    cum_logits = torch.ones(1, 9) * -5.0
    preds = coral_logits_to_predictions(cum_logits)
    assert preds[0] == 0, f"Expected 0, got {preds[0]}"

    # If first 5 logits positive, grade should be 5
    cum_logits = torch.cat([
        torch.ones(1, 5) * 5.0,
        torch.ones(1, 4) * -5.0
    ], dim=1)
    preds = coral_logits_to_predictions(cum_logits)
    assert preds[0] == 5, f"Expected 5, got {preds[0]}"

    print("✅ CORAL predictions: PASSED")

def test_standard_model():
    """Test standard (non-CORAL) model still works."""
    print("\nTesting standard model (for comparison)...")

    model = DualBranchPSA(
        use_coral=False,
        num_classes=10,
        front_depth=18,
        back_depth=18,
        pretrained=False
    )

    batch_size = 4
    front = torch.randn(batch_size, 6, 224, 224)
    back = torch.randn(batch_size, 6, 224, 224)

    outputs = model(front, back)

    # Standard model should output 10 logits
    assert outputs['logits'].shape == (batch_size, 10), f"Expected [4, 10], got {outputs['logits'].shape}"

    print("✅ Standard model: PASSED")

def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\nTesting backward pass...")

    model = DualBranchPSA(
        use_coral=True,
        num_classes=10,
        front_depth=18,
        back_depth=18,
        pretrained=False
    )

    criterion = CompositeLoss(use_coral=True)

    batch_size = 2
    front = torch.randn(batch_size, 6, 224, 224)
    back = torch.randn(batch_size, 6, 224, 224)

    outputs = model(front, back)
    targets = {
        'grade': torch.randint(0, 10, (batch_size,)),
        'edge': torch.randint(0, 2, (batch_size,)).float(),
        'center': torch.randn(batch_size, 2)
    }

    loss_dict = criterion(outputs, targets)
    loss = loss_dict['loss']

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients computed!"

    print("✅ Backward pass: PASSED")

if __name__ == "__main__":
    print("="*60)
    print("CORAL Implementation Test Suite")
    print("="*60)

    try:
        test_coral_model()
        test_coral_loss()
        test_coral_predictions()
        test_standard_model()
        test_backward_pass()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nCORAL implementation is ready for deployment.")
        print("\nTo deploy with CORAL:")
        print("  ./scripts/submit_training.sh")
        print("\nThe script has been updated to include --use_coral flag.")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
