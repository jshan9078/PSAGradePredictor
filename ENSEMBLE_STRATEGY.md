# Ensemble Strategy for PSA Grading Model

## Overview

After three failed single-model optimization attempts (448px: -1.5%, Attention fusion: -4.9%, TTA: -0.43%), we've reached **diminishing returns** for architectural improvements. The ensemble strategy focuses on **model diversity** through different random seeds and configurations.

**Goal:** Improve Val QWK from 0.8359 (Run 7 baseline) to 0.84-0.88 through ensemble averaging.

---

## Why Ensembles Work

### Theory
- **Different models make different errors** - Averaging reduces variance
- **Uncorrelated predictions** - Diversity is key to ensemble success
- **Proven technique** - Industry standard (Kaggle, production ML systems)

### Expected Improvement
- **Conservative estimate:** +0.01-0.02 QWK (0.8359 → 0.85-0.86)
- **Optimistic estimate:** +0.03-0.05 QWK (0.8359 → 0.86-0.88)
- **Best case:** +0.05+ QWK (0.8359 → 0.88+)

Depends on model diversity - more diverse errors = better ensemble.

---

## Ensemble Configuration

### 5-Model Ensemble

We train 5 CORAL models with variations to maximize diversity:

| Model | Seed | Dropout | Weight Decay | Back Depth | Lambda | Purpose |
|-------|------|---------|--------------|------------|---------|---------|
| model1 | 42   | 0.25    | 2e-4        | ResNet-34  | 0.70    | Baseline (Run 7 config) |
| model2 | 123  | 0.20    | 1.5e-4      | ResNet-34  | 0.70    | Lower regularization |
| model3 | 456  | 0.30    | 2.5e-4      | ResNet-34  | 0.70    | Higher regularization |
| model4 | 789  | 0.25    | 2e-4        | ResNet-50  | 0.70    | Deeper backbone |
| model5 | 101  | 0.25    | 2e-4        | ResNet-34  | 0.75    | More weight on back |

### Sources of Diversity

1. **Random seeds** - Different initialization → different local minima
2. **Regularization strength** - Models with different dropout/weight decay learn different patterns
3. **Model capacity** - ResNet-50 has more parameters than ResNet-34
4. **Fusion weight** - λ=0.75 emphasizes back branch more than λ=0.70

### Common Configuration (Constant Across All Models)

```
Loss: CORAL ordinal regression
Image size: 384×384
Batch size: 16
Phase 1: Skipped (phase1_epochs=0)
Phase 2: 20 epochs (models typically peak at epochs 6-15)
Learning rate: 3e-4
Scheduler: ReduceLROnPlateau
Use sampler: True (imbalanced class sampling)
Augmentation: Full (rotate, affine, brightness, blur, noise)
```

---

## Implementation

### 1. Training the Ensemble

**Automated submission script:**
```bash
# Submit all 5 models (parallel training on Vertex AI)
./scripts/submit_ensemble_training.sh --all

# Or submit individual models for testing
./scripts/submit_ensemble_training.sh --model 1
./scripts/submit_ensemble_training.sh --model 2
# ... etc
```

**Training time:**
- ~90 minutes per model on Vertex AI (n1-standard-8 + T4 GPU, 20 epochs)
- Models can train in parallel → total time ~90 minutes
- Estimated cost: ~$4 per model × 5 = $20 total

**Checkpoints saved to:**
```
gs://psa-scan-models-us-east1/ensemble/model1/checkpoints/phase2_best.pth
gs://psa-scan-models-us-east1/ensemble/model2/checkpoints/phase2_best.pth
gs://psa-scan-models-us-east1/ensemble/model3/checkpoints/phase2_best.pth
gs://psa-scan-models-us-east1/ensemble/model4/checkpoints/phase2_best.pth
gs://psa-scan-models-us-east1/ensemble/model5/checkpoints/phase2_best.pth
```

### 2. Evaluating the Ensemble

**After all models finish training:**

```bash
# Evaluate ensemble (auto-discovers all models in ensemble dir)
python src/evaluate_ensemble.py \
  --splits_path data/splits.json \
  --gcs_data_bucket psa-scan-scraping-dataset \
  --ensemble_dir gs://psa-scan-models-us-east1/ensemble/ \
  --eval_individual  # Also show individual model performance

# Alternative: Manually specify checkpoints
python src/evaluate_ensemble.py \
  --splits_path data/splits.json \
  --gcs_data_bucket psa-scan-scraping-dataset \
  --checkpoints \
    gs://bucket/ensemble/model1/checkpoints/phase2_best.pth \
    gs://bucket/ensemble/model2/checkpoints/phase2_best.pth \
    # ... etc
  --eval_individual
```

**Output:**
```
INDIVIDUAL MODEL EVALUATION
============================================================
model1:
  Accuracy: 0.3926
  MAE:      0.6800
  QWK:      0.8350

model2:
  Accuracy: 0.3988
  MAE:      0.6750
  QWK:      0.8280

... (etc for all models)

Individual model QWK statistics:
  Mean:   0.8320
  Std:    0.0050
  Min:    0.8250
  Max:    0.8380

ENSEMBLE EVALUATION
============================================================
Ensemble Results (5 models):
  Accuracy: 0.4110
  MAE:      0.6620
  QWK:      0.8650  ← Target!

IMPROVEMENT ANALYSIS
============================================================
Ensemble QWK:          0.8650
Best individual QWK:   0.8380
Mean individual QWK:   0.8320

Improvement vs best:   +0.0270 (+3.22%)
Improvement vs mean:   +0.0330 (+3.97%)

✅ Ensemble improves over best individual model by 0.0270 QWK!
```

---

## Ensemble Inference Logic

### Prediction Averaging Strategy

For CORAL ordinal regression, we average **cumulative logits** before converting to predictions:

```python
# For each image:
logits_list = []
for model in ensemble:
    outputs = model(front, back)
    logits = outputs['logits']  # [batch, 9] cumulative logits
    logits_list.append(logits)

# Average logits across ensemble
avg_logits = torch.stack(logits_list).mean(dim=0)  # [batch, 9]

# Convert averaged logits to predictions
predictions = coral_logits_to_predictions(avg_logits)
```

**Why average logits instead of predictions?**
- Logits contain more information than discrete predictions
- Averaging predictions loses calibration
- Logit averaging preserves ordinal structure

### Alternative Strategies (Not Implemented)

1. **Weighted averaging:** Give more weight to models with higher validation QWK
2. **Rank averaging:** Average predicted ranks instead of logits
3. **Stacking:** Train a meta-model on ensemble predictions

For our first attempt, simple (unweighted) logit averaging is recommended.

---

## Expected Results

### Success Criteria

**Minimum success:** Ensemble QWK > best individual model QWK
- If best individual is 0.835, ensemble should be ≥ 0.836

**Good success:** Ensemble QWK ≥ 0.85 (+1.7% over Run 7 baseline)
- Validates ensemble approach works for PSA grading

**Great success:** Ensemble QWK ≥ 0.87 (+4.2% over Run 7 baseline)
- Production-ready performance, likely competitive with human graders

### Diversity Analysis

After evaluation, analyze ensemble diversity:

```python
# Disagreement rate between models
for i in range(len(models)):
    for j in range(i+1, len(models)):
        disagreement = (preds_i != preds_j).mean()
        print(f"model{i+1} vs model{j+1}: {disagreement:.2%} disagreement")

# Expected: 15-30% disagreement for good diversity
```

Higher disagreement (with similar individual QWK) → better ensemble potential.

---

## Failure Scenarios & Mitigation

### Scenario 1: Models Too Similar
**Symptom:** All models have nearly identical predictions (disagreement < 10%)
**Cause:** Insufficient diversity in configurations
**Mitigation:**
- Add more seeds (seed=202, 303, 404, ...)
- Vary architecture more (add front_depth=34, back_depth=18)
- Try different optimizers (AdamW with different weight_decay)

### Scenario 2: One Model Much Worse
**Symptom:** One model has QWK < 0.80 (significantly worse than others)
**Cause:** Bad seed, overfitting, or config mismatch
**Mitigation:**
- Exclude that model from ensemble
- Retrain with different seed
- Check training logs for issues

### Scenario 3: Ensemble No Better Than Best Individual
**Symptom:** Ensemble QWK ≤ max(individual QWKs)
**Cause:** Models making same errors (correlated predictions)
**Mitigation:**
- Train more diverse models
- Try different architectures (EfficientNet, Vision Transformer)
- Use different preprocessing (color spaces, edge detection methods)

---

## Next Steps After Ensemble

### If Ensemble Works (QWK ≥ 0.85)
1. **Deploy ensemble to production**
2. **Expand ensemble:** Train 2-3 more models to see if QWK continues improving
3. **Document in CHANGELOG:** Add Section 12 with full results
4. **Ablation study:** Test 3-model vs 4-model vs 5-model ensembles

### If Ensemble Fails (QWK < 0.84)
1. **Analyze diversity:** Are models too similar?
2. **Try different approach:**
   - Self-supervised pretraining on unlabeled cards
   - Multi-scale architecture (combine 384px + 448px features)
   - External data (public card images for pretraining)
3. **Revisit architecture:** Maybe ResNet has plateaued, try EfficientNet or ViT

---

## Cost-Benefit Analysis

### Investment
- **Development time:** 2 hours (scripting + documentation)
- **Training compute:** ~$20 (5 models × $4 each, 20 epochs)
- **Total time:** ~90 minutes (parallel training)

### Expected Return
- **QWK improvement:** +0.02-0.05 (conservative estimate)
- **Business value:** Higher grading accuracy = better trust in AI system
- **Risk:** Low (proven technique, no architectural changes)

**ROI:** High - Ensembles are the most reliable way to improve performance with minimal risk.

---

## References

- **CORAL Paper:** "Rank consistent ordinal regression for neural networks" (Cao et al., 2020)
- **Ensemble Methods:** "Pattern Recognition and Machine Learning" (Bishop, 2006), Chapter 14
- **Kaggle Winning Solutions:** Typically use 5-20 model ensembles
- **Production ML:** Netflix, Google, Amazon all use ensembles for critical prediction tasks

---

## Quick Start

```bash
# 1. Build and push Docker image (if changes made)
./scripts/build_and_push.sh

# 2. Submit ensemble training
./scripts/submit_ensemble_training.sh --all

# 3. Wait ~4 hours for training to complete

# 4. Evaluate ensemble
python src/evaluate_ensemble.py \
  --splits_path data/splits.json \
  --gcs_data_bucket psa-scan-scraping-dataset \
  --ensemble_dir gs://psa-scan-models-us-east1/ensemble/ \
  --eval_individual

# 5. Document results in CHANGELOG.md
```

**Expected outcome:** Val QWK 0.84-0.88, exceeding all previous single-model attempts.
