# Local Testing → Deployment Guide

Complete guide to test locally, then deploy to Vertex AI.

---

## 🧪 Phase 1: Local Testing (Required!)

### Prerequisites

```bash
# Verify you're in the project directory
cd /Users/jonathan/Desktop/psa-estimator

# Activate virtual environment
source venv/bin/activate

# Verify splits.json exists
ls -lh splits.json

# Check you have at least a few sample images locally
# (or the dataset will need to load from GCS)
```

---

### Test 1: Quick Sanity Check (2 minutes)

Run **1 epoch** with **tiny batch** to verify code runs:

```bash
./scripts/test_local.sh
```

**What this tests:**
- ✅ All imports work
- ✅ Dataset loads images
- ✅ Preprocessing works (LAB + CLAHE + gradients)
- ✅ Model initializes
- ✅ Forward pass works
- ✅ Loss computation works
- ✅ Backward pass works
- ✅ Checkpoints save

**Expected output:**
```
Phase 1: Back-Only Pretraining
Epoch 1/1
  Batch 0/X | Loss: X.XXXX (CE: X.XX, EMD: X.XX, Edge: X.XX, Center: X.XX)
  → Saved checkpoint: test_checkpoints/phase1_best.pth

Phase 2: Dual-Branch Fine-Tuning
Epoch 1/1
  Batch 0/X | Loss: X.XXXX
  → Saved checkpoint: test_checkpoints/phase2_best.pth

Training complete! Best QWK: X.XXXX
```

**If it fails:**
- Check error message carefully
- Common issues:
  - Missing dependencies: `pip install -r requirements.txt`
  - Image paths wrong: Verify splits.json paths
  - CUDA issues: Script will auto-fallback to CPU

---

### Test 2: Full Batch Test (Optional, 5 minutes)

Test with **realistic batch size** for 1 epoch:

```bash
python src/train.py \
  --splits_path splits.json \
  --output_dir ./test_checkpoints \
  --batch_size 16 \
  --num_workers 4 \
  --phase1_epochs 1 \
  --phase2_epochs 1 \
  --use_sampler
```

**What this tests:**
- ✅ Imbalanced sampler works
- ✅ Multi-worker data loading
- ✅ Larger batches fit in memory
- ✅ Class weights computed correctly

---

### Test 3: Docker Build Test (Optional, 10 minutes)

Test the Docker container **locally** before pushing:

```bash
# Build image
docker build -t psa-trainer:test .

# Run training in container (CPU-only)
docker run --rm \
  -v $(pwd)/splits.json:/app/splits.json \
  -v $(pwd)/test_checkpoints:/app/checkpoints \
  psa-trainer:test \
  --splits_path /app/splits.json \
  --output_dir /app/checkpoints \
  --batch_size 4 \
  --phase1_epochs 1 \
  --phase2_epochs 1 \
  --no_augment
```

**What this tests:**
- ✅ Docker build succeeds
- ✅ Dependencies installed correctly
- ✅ Container can run training
- ✅ Mounts work correctly

---

## 🚀 Phase 2: Deploy to Vertex AI

Once local tests pass, proceed with deployment:

### Step 1: Upload Data

```bash
./scripts/upload_data.sh
```

**Verifies:**
- `splits.json` uploaded to GCS
- Images accessible in `gs://psa-scan-scraping-dataset/png/`

---

### Step 2: Build & Push Docker Image

```bash
./scripts/build_and_push.sh
```

**What happens:**
1. Builds Docker image with PyTorch 2.4 + CUDA 12.1
2. Tags for Artifact Registry
3. Authenticates Docker with GCP
4. Pushes to `us-east1-docker.pkg.dev/psa-scan-scraping/psa-repo-us-east1/psa-trainer`

**Duration:** ~5-10 minutes

**Common issues:**
- **"Permission denied"**: Run `gcloud auth login`
- **"Repository not found"**: Verify Terraform created artifact registry
- **"Disk space"**: Clean old Docker images: `docker system prune -a`

---

### Step 3: Submit Training Job

```bash
./scripts/submit_training.sh
```

**What happens:**
1. Verifies `splits.json` exists in GCS
2. Submits Vertex AI Custom Training job with:
   - Machine: n1-standard-8 (8 vCPUs, 30 GB RAM)
   - GPU: 1x Tesla T4
   - Image: Latest Docker image
3. Prints job monitoring commands

**Duration:** ~6-8 hours

**Job Configuration:**
```bash
--gcs_data_bucket psa-scan-scraping-dataset     # Images
--splits_path gs://.../splits.json              # Splits
--gcs_checkpoint_dir gs://.../checkpoints/      # Saves here
--gcs_model_dir gs://.../models/                # Final export
--batch_size 32
--phase1_epochs 10
--phase2_epochs 30
```

---

### Step 4: Monitor Training

```bash
# List recent jobs
gcloud ai custom-jobs list \
  --region=us-east1 \
  --filter="displayName:psa-grade-train" \
  --limit=5

# Get latest job ID
JOB_ID=$(gcloud ai custom-jobs list \
  --region=us-east1 \
  --filter="displayName:psa-grade-train" \
  --format="value(name)" \
  --sort-by=~createTime \
  --limit=1)

# Stream logs (real-time)
gcloud ai custom-jobs stream-logs $JOB_ID --region=us-east1
```

**What to watch for:**
- **Class weights**: Should show imbalance (rare grades have higher weights)
- **Loss decreasing**: Both CE and EMD should decrease
- **QWK improving**: Target > 0.85 on validation
- **Checkpoints saving**: Every epoch + best model

**Console UI:**
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=psa-scan-scraping
```

---

### Step 5: Access Results

```bash
# List all checkpoints
gsutil ls gs://psa-scan-models-us-east1/checkpoints/

# Download best model
gsutil cp gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth ./

# Download final exported model
gsutil -m cp -r gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/ ./models/
```

**Files created:**
```
gs://psa-scan-models-us-east1/
├── checkpoints/
│   ├── phase1_best.pth              # Best back-only model
│   ├── phase2_best.pth              # Best dual-branch model (USE THIS!)
│   ├── checkpoint_epoch_5.pth       # Periodic checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── ...
└── models/
    └── psa_dual_branch_v1/
        ├── psa_dual_branch_state_dict.pth   # Model weights
        └── psa_dual_branch_config.json      # Hyperparameters
```

---

## 📊 Expected Training Metrics

### Phase 1: Back-Only Pretraining (10 epochs)
- Initial Loss: ~2.5-3.0
- Final Loss: ~1.8-2.2
- Final QWK: ~0.65-0.75 (back only, not final!)

### Phase 2: Dual-Branch Fine-Tuning (30 epochs)
- Initial Loss: ~1.5-1.8
- Final Loss: ~0.8-1.2
- **Final QWK: ~0.85-0.92** (target!)
- MAE: ~0.3-0.5 grades
- Accuracy: ~75-85%

---

## 🐛 Troubleshooting

### Local Test Issues

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'X'` | `pip install -r requirements.txt` |
| `cv2.imread returned None` | Check image paths in splits.json |
| `RuntimeError: CUDA out of memory` | Reduce `--batch_size` to 4 or 8 |
| `FileNotFoundError: splits.json` | Run `python src/split_dataset.py` first |

### Deployment Issues

| Error | Solution |
|-------|----------|
| Docker build fails | Check Dockerfile syntax, ensure base image accessible |
| Permission denied (GCS) | `gcloud auth login` and `gcloud auth application-default login` |
| Job fails immediately | Check logs: `gcloud ai custom-jobs stream-logs $JOB_ID` |
| `splits.json not found` | Run `./scripts/upload_data.sh` |
| OOM on Vertex AI | Reduce batch size in `submit_training.sh` |

### Training Issues

| Issue | Solution |
|-------|---------|
| Loss not decreasing | Check learning rate, try lower (1e-4) |
| QWK stuck at 0.5 | Model not learning - check data loading |
| NaN loss | Gradient explosion - add gradient clipping (already included) |
| Very slow training | Check GPU utilization: should be >80% |

---

## ✅ Checklist Before Deployment

- [ ] Local test passes (`./scripts/test_local.sh`)
- [ ] `splits.json` generated and verified
- [ ] GCP authentication configured (`gcloud auth login`)
- [ ] Terraform infrastructure deployed (bucket, registry)
- [ ] Images accessible in `gs://psa-scan-scraping-dataset/png/`
- [ ] Docker installed and running
- [ ] Reviewed hyperparameters in `submit_training.sh`

---

## 🎯 Quick Command Reference

```bash
# Local test (REQUIRED first step!)
./scripts/test_local.sh

# Upload data to GCS
./scripts/upload_data.sh

# Build and push Docker image
./scripts/build_and_push.sh

# Submit training job
./scripts/submit_training.sh

# Monitor logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-east1

# Download best model
gsutil cp gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth ./
```

---

## 📈 Next Steps After Training

1. **Evaluate on test set**:
   ```bash
   python src/evaluate.py --checkpoint phase2_best.pth --split test
   ```

2. **Analyze per-grade performance**:
   - Confusion matrix
   - Per-grade MAE
   - Identify which grades are hardest

3. **Deploy for inference**:
   - See `SERVING.md` (to be created)
   - Options: Cloud Run, Vertex AI Endpoints, FastAPI

4. **Iterate**:
   - Adjust hyperparameters based on results
   - Try different lambda_fusion values
   - Experiment with data augmentation

---

## 💰 Cost Reminder

**Local Testing**: FREE!
**Vertex AI Training**: ~$4-5 per run (~6-8 hours @ $0.73/hour)

Always test locally first to catch bugs before spending money! 💸
