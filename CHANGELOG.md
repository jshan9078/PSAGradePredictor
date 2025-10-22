# CHANGELOG - Model Training History

A comprehensive chronicle of all changes, experiments, and optimizations made to the dual-branch PSA card grading model during development. This document records every architectural change, hyperparameter modification, and training strategy evolution in chronological order.

**Current Best Model**: Val QWK 0.7633 @ Epoch 27 (Run #3)
**Last Updated**: October 21, 2025
**Status**: Testing incremental improvements (label smoothing)

---

## Table of Contents
1. [Preprocessing Performance Fix](#1-preprocessing-performance-fix)
2. [Validation Loss Calculation Bug](#2-validation-loss-calculation-bug)
3. [Learning Rate Scheduler Fix](#3-learning-rate-scheduler-fix)
4. [Overfitting Mitigation Attempt 1](#4-overfitting-mitigation-attempt-1)
5. [Training Strategy Redesign](#5-training-strategy-redesign)
6. [Loss Reduction Strategies - Failed Attempt](#6-loss-reduction-strategies---failed-attempt)
7. [Incremental Approach - Label Smoothing Only (Current)](#7-incremental-approach---label-smoothing-only-current)

---

## 1. Preprocessing Performance Fix

### Problem
- Training was **extremely slow** (2+ minutes per batch)
- Preprocessing bottleneck identified in LAB color space conversion and gradient computation

### Root Cause
- Using `scikit-image` filters (`sobel_h`, `sobel_v`, `laplace`) which are 10-100x slower than OpenCV
- Preprocessing happening on CPU during training, blocking GPU

### Solution
**File:** `src/preprocess.py`

Replaced scikit-image with OpenCV implementations:
```python
# Before (slow)
from skimage.filters import laplace, sobel_h, sobel_v
gx = sobel_h(L_eq)
gy = sobel_v(L_eq)
lap = laplace(L_eq)

# After (fast)
import cv2
gx = cv2.Sobel(L_eq, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(L_eq, cv2.CV_64F, 0, 1, ksize=3)
lap = cv2.Laplacian(L_eq, cv2.CV_64F)
gx = np.abs(gx)
gy = np.abs(gy)
lap = np.abs(lap)
```

### Result
- ✅ **10-50x speedup** in preprocessing
- ✅ Preprocessing no longer a bottleneck
- ✅ Training progressed to reveal other issues

---

## 2. Validation Loss Calculation Bug

### Problem
- **Validation loss showing unrealistic values** (281-648) while training loss was ~2.3
- Indicated fundamental calculation error

### Investigation Timeline

#### Attempt 1: Division Error (Failed)
Initially thought the issue was batch size division:
```python
# Wrong fix - made it worse (val loss = 2944)
total_loss += loss_dict['loss'].item() * batch_size
```

#### Attempt 2: Correct Division (Partial Fix)
Fixed to proper averaging:
```python
# Better but still wrong (val loss = 461 → 281)
total_loss += loss_dict['loss'].item()
return total_loss / num_batches
```

#### Root Cause Discovery: Phase Mismatch
**The real bug:** During Phase 1 (back-only training):
- **Training** passed `torch.zeros_like(front)` (dummy fronts)
- **Validation** passed real front images to the **frozen front branch**

The frozen front encoder output garbage on real images, causing massive loss.

### Solution
**File:** `src/train.py` - `validate()` function

Added `phase` parameter to match training behavior:
```python
def validate(model, loader, criterion, device, phase="dual"):
    # ...
    if phase == "back_only":
        dummy_front = torch.zeros_like(front)
        outputs = model(dummy_front, back)
    else:
        outputs = model(front, back)
```

### Result
- ✅ Validation loss now realistic (~2.7 vs training ~2.4)
- ✅ Training could finally proceed to completion
- ✅ Revealed overfitting problem (next section)

---

## 3. Learning Rate Scheduler Fix

### Problem
- **Validation loss oscillating** (2.02 → 2.43 → 2.15 → 2.43)
- No "Reducing learning rate" messages in logs
- Learning rate not adapting to training progress

### Root Cause
1. Phase 2 was using `OneCycleLR` scheduler
2. **No `scheduler.step()` call** in the Phase 2 training loop
3. Learning rate stayed constant, causing oscillations

### Solution
**File:** `src/train.py`

Changed scheduler and added step call:
```python
# Before: OneCycleLR (cyclic LR)
scheduler = OneCycleLR(optimizer, max_lr=args.lr_phase2, ...)

# After: ReduceLROnPlateau (adaptive LR)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',           # Maximize QWK
    factor=0.5,           # Reduce LR by 50%
    patience=3,           # Wait 3 epochs
    verbose=True,
    min_lr=1e-6
)

# Added in training loop:
scheduler.step(val_metrics['qwk'])
```

Also reduced Phase 2 learning rate from `3e-4` to `1e-4` for more stability.

### Result
- ✅ Learning rate now adapts to plateau
- ✅ More stable training dynamics
- ✅ Allowed training to proceed further

---

## 4. Overfitting Mitigation Attempt 1

### Problem
At epoch 40 of first successful run:
```
Train Loss: 0.18, Train QWK: 0.95
Val Loss: 6.36, Val QWK: 0.47
```
- **35x loss gap** between train and validation
- **2x QWK gap** - model memorizing training data
- 33M parameters for 9,824 images = **3,365 params/image** (too high capacity)

### Solution Attempt: Aggressive Regularization
**Files:** `scripts/submit_training.sh`, model configuration

Changed:
```
Model:
  - Front: ResNet-34 → ResNet-18
  - Back: ResNet-34 → ResNet-18
  - Total params: 33M → ~20M

Regularization:
  - Dropout: 0.1 → 0.4
  - Weight decay: 1e-4 → 5e-4
```

### Result
- ❌ **Val QWK: 0.526** at epoch 15 (worse than previous 0.56)
- ❌ **Over-regularization** - model too constrained
- ❌ Hurt performance instead of improving it

### Lessons Learned
- Too much regularization prevents learning
- Need balanced approach
- Model capacity wasn't the only issue

---

## 5. Training Strategy Redesign

### Motivation
User insight: *"For Pokemon cards, the back alone is typically enough to estimate PSA and the front either seals the deal or invalidates it."*

This suggested the two-phase curriculum learning (back-only → dual-branch) wasn't helping.

### Hypothesis
- Back-only pretraining not contributing to final performance
- Dual-branch learning from start might work better
- Previous Phase 1 may have been wasted compute

### Solution
**File:** `scripts/submit_training.sh`

```bash
# Configuration changes:
--phase1_epochs=0           # Skip back-only training entirely
--phase2_epochs=50          # Train dual-branch from start
--front_depth=18            # Smaller front encoder
--back_depth=34             # Restore larger back encoder
--dropout=0.25              # Moderate regularization
--weight_decay=2e-4         # Moderate L2 penalty
--lr_phase2=3e-4            # Return to higher LR
```

### Result (Epoch 27 - Best)
```
Train Loss: 0.49, Train QWK: 0.87
Val Loss: 3.53, Val QWK: 0.76
```

- ✅ **Val QWK: 0.7633** - exceeded 0.7+ goal!
- ✅ Dual-branch from start strategy validated
- ⚠️ Still shows overfitting (7.2x loss gap)
- ⚠️ Validation loss high despite good QWK

### Why It Worked
1. Front and back branches learned to work together from start
2. No wasted epochs on frozen front encoder
3. Better balance between model capacity and regularization
4. More training epochs (50 vs 30) allowed learning

---

## 6. Loss Reduction Strategies - Failed Attempt

### Problem
Current best model (Epoch 27):
```
Val Loss: 3.53 (high)
Val QWK: 0.76 (good)
Train Loss: 0.49 (7.2x gap = overfitting)
```

**Question:** Why is validation loss high when QWK is good?

**Answer:**
- **QWK** measures ranking agreement (forgiving of near-misses)
- **Cross-Entropy Loss** heavily penalizes any wrong prediction
- **Class weights** amplify loss (Grade 2 has 42.67x weight)
- **EMD loss** is strict on probability distributions
- **Overconfident predictions** (from overfitting) cause high CE loss

### Four-Pronged Solution

#### Strategy 1: Label Smoothing
**File:** `src/losses.py` - `WeightedCrossEntropy`

```python
# Before: Hard targets [0,0,0,0,1,0,0,0,0,0]
# After: Soft targets [0.02,0.02,0.02,0.02,0.84,0.02,0.02,0.02,0.02,0.02]

class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight=self.w,
            label_smoothing=self.label_smoothing  # NEW
        )
```

**Why:** Prevents overconfident predictions, reduces CE loss without hurting QWK

**Expected Impact:** Reduce val loss by 30-50%, maintain or improve QWK

**Implementation:** `--label_smoothing=0.1`

---

#### Strategy 2: Cap Class Weights
**File:** `src/train.py` - `compute_class_weights()`

```python
# Before: Unlimited weights (Grade 2 = 42.67x)
raw_weight = total / (num_classes * count)

# After: Capped at 10x
weights[class_idx] = min(10.0, raw_weight)
```

**Why:** 42.67x weight causes huge loss spikes on single Grade 2/3 misclassifications

**Expected Impact:** Reduce val loss by 20-30%, more stable training

**Trade-off:** May slightly hurt rare class recall (acceptable)

---

#### Strategy 3: Mixup Augmentation
**File:** `src/train.py` - New functions

```python
def mixup_data(front, back, targets_dict, alpha=0.4):
    """Mix two training samples and their labels."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size)

    mixed_front = lam * front + (1 - lam) * front[index]
    mixed_back = lam * back + (1 - lam) * back[index]

    return mixed_front, mixed_back, targets_a, targets_b, lam

def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """Compute mixed loss."""
    loss_a = criterion(outputs, targets_a)
    loss_b = criterion(outputs, targets_b)
    return lam * loss_a + (1 - lam) * loss_b
```

**How it works:**
- Mixes two images: `new_img = 0.7 * img1 + 0.3 * img2`
- Mixes their labels: `new_grade = 0.7 * grade1 + 0.3 * grade2`
- Forces model to learn smooth interpolations

**Why:**
- Regularizes model, prevents overfitting
- Improves generalization
- Works especially well for ordinal problems (grades are sequential)

**Expected Impact:** Reduce train/val gap significantly, lower val loss 25-35%, potentially improve QWK to 0.78-0.80

**Implementation:** `--mixup_alpha=0.3`

---

#### Strategy 4: Reduce EMD Coefficient
**File:** `src/losses.py` - `CompositeLoss`

```python
# Before:
loss = CE + 0.7*EMD + 0.05*edge + 0.1*center

# After:
loss = CE + 0.4*EMD + 0.05*edge + 0.1*center
```

**Why:** EMD is the strictest loss component, reducing its weight helps

**Expected Impact:** Directly reduce val loss by 10-20%

**Implementation:** `--alpha_emd=0.4` (default changed from 0.7)

---

### Combined Expected Results

**Conservative estimate:**
- Val Loss: **3.53 → 2.2-2.5** (30-35% reduction)
- Val QWK: **0.76 → 0.77-0.80** (maintained or improved)
- Train/Val Gap: **7.2x → 4-5x** (reduced overfitting)

**Optimistic estimate:**
- Val Loss: **3.53 → 1.8-2.2** (40-50% reduction)
- Val QWK: **0.76 → 0.78-0.82** (improved generalization)
- Train/Val Gap: **7.2x → 3-4x** (significantly reduced)

---

## Summary of All Hyperparameter Changes

### Original Configuration (Failed - Overfitting)
```
Model: ResNet-34 front, ResNet-34 back
Phase 1: 10 epochs (back-only)
Phase 2: 30 epochs
LR Phase 2: 3e-4
Dropout: 0.1
Weight decay: 1e-4
Alpha EMD: 0.7
Label smoothing: 0.0
Mixup: Disabled
Class weights: Unlimited
```
**Result:** Train QWK 0.95, Val QWK 0.47 (severe overfitting)

---

### Over-Regularized Configuration (Failed - Too Constrained)
```
Model: ResNet-18 front, ResNet-18 back
Phase 1: 10 epochs
Phase 2: 30 epochs
LR Phase 2: 1e-4
Dropout: 0.4
Weight decay: 5e-4
Alpha EMD: 0.7
Label smoothing: 0.0
Mixup: Disabled
Class weights: Unlimited
```
**Result:** Val QWK 0.526 at epoch 15 (worse performance)

---

### Successful Configuration (Current Best)
```
Model: ResNet-18 front, ResNet-34 back
Phase 1: 0 epochs (skip)
Phase 2: 50 epochs
LR Phase 2: 3e-4
Dropout: 0.25
Weight decay: 2e-4
Alpha EMD: 0.7
Label smoothing: 0.0
Mixup: Disabled
Class weights: Unlimited
Scheduler: ReduceLROnPlateau (patience=3)
```
**Result:** Train QWK 0.87, Val QWK 0.76 (achieved goal, but high val loss)

---

### All Strategies Combined (Failed - Over-Regularized)
```
Model: ResNet-18 front, ResNet-34 back
Phase 1: 0 epochs (skip)
Phase 2: 50 epochs
LR Phase 2: 3e-4
Dropout: 0.25
Weight decay: 2e-4
Alpha EMD: 0.4              ← REDUCED
Label smoothing: 0.1        ← NEW
Mixup alpha: 0.3            ← NEW
Class weights: Capped at 10x ← NEW
Scheduler: ReduceLROnPlateau (patience=3)
```
**Result:** Val QWK 0.287 at epoch 13 (WORST EVER - stopped immediately)

**Why it failed:**
- Compounding regularization effect
- 4 strategies simultaneously prevented effective learning
- Mixup (0.3) + label smoothing (0.1) + reduced EMD (0.4) + capped weights = too constrained
- Model couldn't learn ordinal structure effectively
- Similar to previous dropout 0.4 failure

**Lesson learned:** Test one strategy at a time, not all together

---

## 7. Incremental Approach - Label Smoothing Only (Current)

### Strategy Change
After the failure of combining all 4 strategies, pivoted to **incremental testing** - deploying one strategy at a time to isolate effects.

### Rationale for Label Smoothing First
1. **Lowest risk** - industry standard technique
2. **Directly addresses high val loss** - prevents overconfident wrong predictions
3. **No interaction effects** - pure loss function change
4. **Fast feedback** - should see improvements by epoch 15

### Configuration
```
Model: ResNet-18 front, ResNet-34 back
Phase 1: 0 epochs (skip)
Phase 2: 50 epochs
LR Phase 2: 3e-4
Dropout: 0.25
Weight decay: 2e-4
Alpha EMD: 0.7              ← RESTORED to baseline
Label smoothing: 0.1        ← ONLY change from baseline
Mixup alpha: 0.0            ← DISABLED
Class weights: Unlimited    ← RESTORED (no cap)
Scheduler: ReduceLROnPlateau (patience=3)
```

### Expected Results

**Conservative estimate (most likely):**
- Val QWK: **0.74-0.76** (maintain baseline performance)
- Val Loss: **3.0-3.3** (reduction from 3.5)
- By epoch 13: Val QWK ~0.50-0.55 (healthy learning curve)

**Optimistic estimate:**
- Val QWK: **0.76-0.78** (slight improvement from smoother gradients)
- Val Loss: **2.8-3.2** (30% reduction)
- By epoch 13: Val QWK ~0.55-0.60

**Failure threshold:**
- If Val QWK < 0.40 at epoch 13 → even label smoothing alone is too much
- If Val QWK < 0.70 by epoch 27 → label smoothing not helping

### What Label Smoothing Does

**Mathematically:**
```python
# Hard labels (original):
Target for grade 8: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

# Soft labels (smoothing=0.1):
Target for grade 8: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01]
```

**Effect on training:**
- Model can't reach 100% confidence even when correct
- Cross-entropy loss is lower for slightly uncertain predictions
- Prevents catastrophic loss spikes from overconfident errors
- Smooths gradient updates

**Why it should help with validation loss:**
Current problem: Model outputs `P(grade 8) = 0.98` but truth is grade 9
- **Without smoothing:** CE loss ≈ -log(0.02) = 3.91 (huge penalty)
- **With smoothing:** Softer penalties, lower loss even when wrong

### Future Plans (If Successful)

**Next experiment (Run after this succeeds):**
Test adding ONE more strategy:
```
Option A: Label smoothing (0.1) + Capped weights (10x)
Option B: Label smoothing (0.1) + Reduced EMD (0.5)
```

**Avoid for now:**
- Mixup - too aggressive, slows learning significantly
- Can reconsider mixup only if overfitting persists after other fixes

### Files Modified
- `src/losses.py` - Added `label_smoothing` parameter to `WeightedCrossEntropy` and `CompositeLoss`
- `src/train.py` - Reverted class weight capping (back to unlimited)
- `scripts/submit_training.sh` - Updated args: `--alpha_emd=0.7`, `--label_smoothing=0.1`, `--mixup_alpha=0.0`
- `src/augmentations.py` - Fixed `GaussNoise` parameter (`var_limit` → `variance_limit`)

### Status
- ⏳ Ready to deploy after Docker rebuild
- 📊 Monitoring checkpoint: Epoch 13 (should show Val QWK ~0.50-0.55)
- 🎯 Success criteria: Val QWK ≥0.74, Val Loss ≤3.3 by epoch 27

---

## Key Insights Learned

### 1. Preprocessing Matters
- **10-50x speedups** possible by choosing right libraries (OpenCV vs scikit-image)
- CPU preprocessing can be a hidden bottleneck

### 2. Validation Must Match Training
- Phase mismatch bugs are subtle but catastrophic
- Always verify validation uses same data augmentation/preprocessing as training

### 3. Learning Rate Scheduling is Critical
- Fixed learning rates cause oscillation
- Adaptive scheduling (ReduceLROnPlateau) handles plateaus automatically
- Must actually call `scheduler.step()`!

### 4. Balance Regularization
- Too little → overfitting (QWK 0.95 train, 0.47 val)
- Too much → underfitting (QWK 0.526 val)
- Sweet spot: dropout 0.25, weight decay 2e-4

### 5. Domain Knowledge Guides Architecture
- User insight about Pokemon cards led to skipping Phase 1
- Saved compute time and improved performance
- **Lesson:** Always leverage domain expertise

### 6. Loss ≠ Performance
- Can have high validation loss (3.53) but good QWK (0.76)
- Different metrics measure different things
- QWK is primary metric for ordinal classification

### 7. Compounding Regularization is Dangerous ⚠️
- **Multiple regularization strategies don't add linearly - they multiply**
- Combining 4 strategies (label smoothing + mixup + reduced EMD + capped weights) → Val QWK 0.287 (worst ever)
- Similar to dropout 0.4 failure (Val QWK 0.526)
- **Always test strategies incrementally**, one at a time
- Establish baseline → add one change → measure → repeat

### 8. Class Imbalance Requires Care
- Extreme weights (42.67x) cause unstable training
- Capping weights balances rare class learning with stability (but can hurt performance when combined with other regularization)
- Label smoothing helps with overconfidence from imbalance

### 9. Mixup Can Slow Learning Significantly
- Mixup alpha=0.3 dramatically slowed early epoch learning
- Benefits may appear later (epochs 20-30) but high risk
- Best reserved for cases with severe overfitting
- Not needed if label smoothing + other techniques work

---

## Next Steps

### Immediate (Current Run)
1. ✅ Reverted to incremental approach - label smoothing ONLY
2. ⏳ Rebuild Docker image with reverted changes
3. ⏳ Deploy to Vertex AI
4. ⏳ Monitor epoch 13 (target: Val QWK ~0.50-0.55)
5. ⏳ Monitor epoch 27 (target: Val QWK ≥0.74, Val Loss ≤3.3)

### If Label Smoothing Succeeds (Val QWK ≥0.74, Loss ≤3.3)
**Next test (one additional strategy):**
- Option A: Add capped weights (10x max)
- Option B: Add reduced EMD coefficient (0.5 or 0.6)
- Option C: Increase label smoothing to 0.15

**Then:**
- Test on holdout test set
- Analyze per-grade performance (confusion matrix)
- Consider ensemble methods if needed
- Deploy to production

### If Label Smoothing Fails (Val QWK < 0.70 or Loss > 3.4)
- Return to pure baseline (no label smoothing)
- Accept Val QWK 0.76 with Val Loss 3.5
- Focus on other improvements:
  - Collect more data for rare grades
  - Try focal loss
  - Experiment with different model architectures
  - Test larger image sizes (448px)

### Advanced Optimizations (Future)
- Temperature scaling for better calibration
- Test-time augmentation (TTA)
- Knowledge distillation
- Collect more data for rare grades (2, 3, 10)

---

## File Change Summary

### Modified Files
1. **src/preprocess.py** - OpenCV speedup
2. **src/train.py** - Validation phase fix, scheduler fix, mixup, class weight cap, LR logging
3. **src/losses.py** - Label smoothing, EMD coefficient default
4. **scripts/submit_training.sh** - Hyperparameter updates

### New Features Added
- Label smoothing (configurable via argument)
- Mixup augmentation (configurable, currently disabled)
- Class weight capping capability (currently reverted to unlimited)
- Learning rate logging
- Phase-aware validation

### Arguments Added
- `--label_smoothing` (default: 0.1, currently active)
- `--mixup_alpha` (default: 0.3, currently set to 0.0)
- `--alpha_emd` (default: 0.4, currently reverted to 0.7)

---

## Training Run Summary

| Run | Configuration | Epoch 13 Val QWK | Best Val QWK | Result |
|-----|--------------|------------------|--------------|--------|
| 1 | Original (ResNet-34/34, dropout 0.1) | ~0.45 | 0.47 @ epoch 40 | ❌ Severe overfitting |
| 2 | Over-regularized (ResNet-18/18, dropout 0.4) | - | 0.526 @ epoch 15 | ❌ Too constrained |
| 3 | Dual-branch from start (baseline) | ~0.55 | **0.7633 @ epoch 27** | ✅ **Best so far** |
| 4 | All 4 strategies combined | **0.287** | - | ❌ **Worst ever** - stopped |
| 5 | Label smoothing only (current) | TBD | TBD | ⏳ **Testing now** |

---

**Document Version:** 2.0
**Last Updated:** 2025-10-21
**Current Best Model:** Epoch 27 from Run 3 (Val QWK 0.7633, Val Loss 3.53)
**Status:** Testing label smoothing only (incremental approach) after combined strategies failed
**Next Milestone:** Epoch 13 of current run (target: Val QWK ~0.50-0.55)
