# Phase 4 Quickstart: Train on Vertex AI

Complete end-to-end guide to train your PSA grading model on Google Cloud.

## Prerequisites Checklist

- [ ] Google Cloud project: `psa-scan-scraping`
- [ ] Terraform infrastructure deployed (bucket, artifact registry)
- [ ] `gcloud` CLI installed and authenticated
- [ ] Docker installed
- [ ] `splits.json` generated locally

## 5-Step Deployment

### 1️⃣ Upload Data to GCS

```bash
./scripts/upload_data.sh
```

**What it does**: Uploads `splits.json` to `gs://psa-scan-models-us-east1/data/`

---

### 2️⃣ Build & Push Docker Image

```bash
./scripts/build_and_push.sh
```

**What it does**:
- Builds Docker image with PyTorch 2.4 + CUDA 12.1
- Pushes to Artifact Registry: `us-east1-docker.pkg.dev/psa-scan-scraping/psa-repo-us-east1/psa-trainer`

**Duration**: ~5-10 minutes

---

### 3️⃣ Submit Training Job

```bash
./scripts/submit_training.sh
```

**What it does**:
- Submits Vertex AI custom training job
- **Machine**: n1-standard-8 (8 vCPUs, 30 GB RAM)
- **GPU**: 1x NVIDIA Tesla T4
- **Training**: Phase 1 (10 epochs) + Phase 2 (30 epochs)

**Duration**: ~6-8 hours

---

### 4️⃣ Monitor Training

```bash
# View job status
gcloud ai custom-jobs list --region=us-east1 --filter="displayName:psa-grade-train" --limit=5

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-east1
```

**Console**: https://console.cloud.google.com/vertex-ai/training/custom-jobs

---

### 5️⃣ Access Outputs

```bash
# List checkpoints
gsutil ls gs://psa-scan-models-us-east1/checkpoints/

# Download best model
gsutil cp gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth ./

# Download final model export
gsutil -m cp -r gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/ ./models/
```

**Outputs**:
- Checkpoints: `gs://psa-scan-models-us-east1/checkpoints/`
  - `phase1_best.pth`
  - `phase2_best.pth`
  - `checkpoint_epoch_N.pth`
- Final model: `gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/`
  - `psa_dual_branch_state_dict.pth`
  - `psa_dual_branch_config.json`

---

## Training Configuration

| Setting | Value | Modify In |
|---------|-------|-----------|
| Batch size | 32 | `submit_training.sh` |
| Phase 1 epochs | 10 | `submit_training.sh` |
| Phase 2 epochs | 30 | `submit_training.sh` |
| Learning rate (Phase 1) | 1e-3 | `submit_training.sh` |
| Learning rate (Phase 2) | 3e-4 | `submit_training.sh` |
| GPU type | Tesla T4 | `submit_training.sh` |
| Machine type | n1-standard-8 | `submit_training.sh` |

---

## Cost Estimate

**Per Training Run**:
- Compute: ~$0.73/hour × 6 hours = **$4.40**
- Storage: Negligible (~$0.10/month)
- **Total**: ~$5 per training run

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "splits.json not found" | Run `./scripts/upload_data.sh` |
| "Image not found" | Run `./scripts/build_and_push.sh` |
| "Permission denied" | Run `gcloud auth login` and `gcloud auth application-default login` |
| "Out of memory" | Reduce `--batch_size` in `submit_training.sh` |

---

## File Structure

```
psa-estimator/
├── Dockerfile                      # Training container
├── requirements.txt                # Python dependencies
├── splits.json                     # Train/val/test splits
├── src/
│   ├── train.py                    # Main training script
│   ├── model.py                    # Dual-branch architecture
│   ├── losses.py                   # Composite loss function
│   ├── dataset.py                  # PSA dataset loader
│   ├── preprocess.py               # LAB + CLAHE + gradients
│   ├── gcs_utils.py                # GCS checkpoint saving
│   ├── sampler.py                  # Imbalanced sampler
│   └── augmentations.py            # Data augmentation
└── scripts/
    ├── upload_data.sh              # Upload data to GCS
    ├── build_and_push.sh           # Build Docker image
    └── submit_training.sh          # Submit training job
```

---

## Next Steps After Training

1. **Download checkpoints**:
   ```bash
   gsutil cp gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth ./
   ```

2. **Evaluate model**:
   ```bash
   python src/evaluate.py --checkpoint phase2_best.pth
   ```

3. **Deploy for inference**:
   - Cloud Run (serverless API)
   - Vertex AI Endpoints
   - Custom FastAPI server

---

## Questions?

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed documentation.
