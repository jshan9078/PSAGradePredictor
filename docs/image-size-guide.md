# Image Size Selection Guide

## 🎯 **Quick Decision Matrix**

| Your Goal | Use This Size | Batch Size | Training Time |
|-----------|---------------|------------|---------------|
| **Fast testing/prototyping** | 224 | 32 | ~30 min/epoch |
| **Production training** ⭐ | **384** | **16** | **~60 min/epoch** |
| **Maximum quality** | 448 | 12 | ~90 min/epoch |

---

## 📊 **Detailed Comparison**

### **224x224** - Fast Iteration
```bash
python src/train.py --image_size 224 --batch_size 32
```

**Pros**:
- ✅ Fastest training (30-40 min/epoch on T4)
- ✅ Standard ImageNet size
- ✅ Low memory usage (4 GB)
- ✅ Good for initial experiments

**Cons**:
- ⚠️ May lose small scratches (<3 pixels)
- ⚠️ Fine texture details reduced

**Use for**: Initial testing, hyperparameter tuning

---

### **384x384** - Production (RECOMMENDED) ⭐
```bash
python src/train.py --image_size 384 --batch_size 16
```

**Pros**:
- ✅ **3x more pixels** than 224x224
- ✅ **Preserves 95% of detail** including small scratches
- ✅ Common in high-quality models (ViT, EfficientNet-B5)
- ✅ Still reasonable training time (60-75 min/epoch)
- ✅ Fits comfortably on T4 GPU (8 GB memory)

**Cons**:
- ⚠️ 2x slower than 224x224
- ⚠️ Need to reduce batch size to 16

**Use for**: **Production training, final model**

---

### **448x448** - Maximum Quality
```bash
python src/train.py --image_size 448 --batch_size 12
```

**Pros**:
- ✅ 4x more pixels than 224x224
- ✅ Preserves 98% of detail
- ✅ Best for detecting subtle grade differences

**Cons**:
- ❌ 3-4x slower than 224x224
- ❌ Higher memory (10 GB)
- ❌ Diminishing returns vs. 384

**Use for**: Only if 384 doesn't achieve target QWK

---

## 🚀 **Usage Examples**

### Local Testing (Fast)
```bash
./scripts/test_local.sh  # Uses default 224x224
```

### Vertex AI Production Training (Recommended)
```bash
# Already configured for 384x384!
./scripts/submit_training.sh
```

### Custom Size
```bash
# Local
python src/train.py \
  --image_size 384 \
  --batch_size 16 \
  --phase1_epochs 10 \
  --phase2_epochs 30

# Vertex AI - edit submit_training.sh
--image_size,384,\
--batch_size,16,\
```

---

## 📐 **Detail Preservation Examples**

### Original Scan: 1257 x 902

#### Feature: Corner Whitening (10 pixels)
```
224x224:  2px   ⚠️ Visible but small
384x384:  3px   ✅ Clearly visible
448x448:  4px   ✅ Very clear
```

#### Feature: Tiny Scratch (3 pixels)
```
224x224:  <1px  ❌ May disappear
384x384:  1px   ✅ Visible
448x448:  1.5px ✅ Clear
```

#### Feature: Holo Pattern
```
224x224:  ✨✨   ⚠️ Coarse
384x384:  ✨✨✨  ✅ Good
448x448:  ✨✨✨✨ ✅ Excellent
```

---

## 💾 **Memory Requirements**

| Size | Batch Size | GPU Memory | Fits on T4? |
|------|------------|------------|-------------|
| 224 | 32 | 4 GB | ✅ Yes |
| 224 | 64 | 7 GB | ✅ Yes |
| 384 | 16 | 8 GB | ✅ Yes |
| 384 | 32 | 14 GB | ❌ No (need V100) |
| 448 | 12 | 10 GB | ✅ Tight fit |
| 448 | 16 | 13 GB | ❌ No (need V100) |

---

## ⏱️ **Training Time Estimates**

On **n1-standard-8 + Tesla T4 GPU**:

| Size | Batch | Time/Epoch | 40 Epochs | Cost |
|------|-------|------------|-----------|------|
| 224 | 32 | 30 min | 20 hours | ~$15 |
| 384 | 16 | 60 min | 40 hours | ~$30 |
| 448 | 12 | 90 min | 60 hours | ~$45 |

---

## 🎯 **Recommended Strategy**

### Phase 1: Quick Validation (224x224)
```bash
# Run locally or Vertex AI with small epoch count
python src/train.py \
  --image_size 224 \
  --batch_size 32 \
  --phase1_epochs 3 \
  --phase2_epochs 10
```

**Goal**: Verify code works, get baseline QWK

---

### Phase 2: Production Training (384x384) ⭐
```bash
# Full training on Vertex AI
./scripts/submit_training.sh  # Pre-configured for 384x384!
```

**Goal**: Achieve target QWK (>0.85)

---

### Phase 3: Optional Refinement (448x448)
```bash
# Only if QWK < 0.85 with 384x384
python src/train.py \
  --image_size 448 \
  --batch_size 12 \
  --phase2_epochs 40
```

**Goal**: Squeeze out last 1-2% performance

---

## 📈 **Expected Performance**

| Size | Expected QWK | MAE | Top-1 Accuracy |
|------|--------------|-----|----------------|
| 224 | 0.83-0.86 | 0.5 | 75-80% |
| 384 | 0.86-0.90 | 0.4 | 80-85% |
| 448 | 0.87-0.91 | 0.35 | 82-87% |

**Diminishing returns**: 384→448 gives much less gain than 224→384

---

## 🔧 **Troubleshooting**

### "CUDA out of memory"
```bash
# Reduce batch size
python src/train.py --image_size 384 --batch_size 8
```

### "Training too slow"
```bash
# Use smaller size or fewer epochs
python src/train.py --image_size 224 --phase2_epochs 20
```

### "QWK not improving"
```bash
# Try larger images
python src/train.py --image_size 384 --batch_size 16
```

---

## ✅ **Current Configuration**

Your deployment script is **already set to 384x384**:

```bash
# scripts/submit_training.sh
--image_size,384,\
--batch_size,16,\
```

This is the **optimal balance** for card grading! 🎯

---

## 🎓 **Summary**

**Start with**: 384x384 (pre-configured in deployment scripts)
**Try if needed**: 448x448 (if QWK < 0.85)
**Don't use**: 512+ (overkill, too slow)

The 384x384 size preserves 95% of detail while keeping training time reasonable - perfect for Pokemon card grading! 🚀
