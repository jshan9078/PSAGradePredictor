# Phase 4: Training on Vertex AI

This guide walks through deploying your PSA grading model training to Google Cloud Vertex AI.

## Prerequisites

1. **Google Cloud Project**: `psa-scan-scraping`
2. **Terraform resources created**:
   - Storage bucket: `psa-scan-models-us-east1`
   - Artifact Registry: `psa-repo-us-east1`
   - Vertex AI Workbench instance (optional for development)

3. **Local tools installed**:
   ```bash
   gcloud --version  # Google Cloud SDK
   docker --version  # Docker Engine
   ```

4. **Authentication**:
   ```bash
   gcloud auth login
   gcloud config set project psa-scan-scraping
   gcloud auth application-default login
   ```

---

## Step 1: Prepare Training Data

Upload your splits.json to GCS:

```bash
gsutil cp splits.json gs://psa-scan-models-us-east1/data/
```

Verify upload:
```bash
gsutil ls gs://psa-scan-models-us-east1/data/splits.json
```

---

## Step 2: Build and Push Docker Image

The training code is packaged as a Docker container with PyTorch 2.4 + CUDA 12.1.

### Option A: Automated Script

```bash
cd /Users/jonathan/Desktop/psa-estimator
./scripts/build_and_push.sh
```

### Option B: Manual Steps

```bash
# Build image
docker build -t psa-trainer:latest .

# Tag for Artifact Registry
IMAGE_URI="us-east1-docker.pkg.dev/psa-scan-scraping/psa-repo-us-east1/psa-trainer:latest"
docker tag psa-trainer:latest ${IMAGE_URI}

# Configure Docker auth
gcloud auth configure-docker us-east1-docker.pkg.dev

# Push to Artifact Registry
docker push ${IMAGE_URI}
```

---

## Step 3: Submit Training Job to Vertex AI

### Option A: Automated Script

```bash
./scripts/submit_training.sh
```

### Option B: Manual Submission

```bash
JOB_NAME="psa-grade-train-$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="us-east1-docker.pkg.dev/psa-scan-scraping/psa-repo-us-east1/psa-trainer:latest"

gcloud ai custom-jobs create \
  --region=us-east1 \
  --display-name=${JOB_NAME} \
  --project=psa-scan-scraping \
  --worker-pool-spec=\
machine-type=n1-standard-8,\
replica-count=1,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
container-image-uri=${IMAGE_URI} \
  --args="\
--splits_path,gs://psa-scan-models-us-east1/data/splits.json,\
--output_dir,/tmp/checkpoints,\
--gcs_checkpoint_dir,gs://psa-scan-models-us-east1/checkpoints/,\
--gcs_model_dir,gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/,\
--batch_size,32,\
--phase1_epochs,10,\
--phase2_epochs,30,\
--lr_phase1,1e-3,\
--lr_phase2,3e-4,\
--use_sampler"
```

---

## Step 4: Monitor Training

### View Job Status

```bash
gcloud ai custom-jobs list \
  --region=us-east1 \
  --filter="displayName:psa-grade-train" \
  --sort-by=~createTime \
  --limit=5
```

### Stream Logs

```bash
# Get job ID
JOB_ID=$(gcloud ai custom-jobs list \
  --region=us-east1 \
  --filter="displayName:psa-grade-train" \
  --format="value(name)" \
  --limit=1)

# Stream logs
gcloud ai custom-jobs stream-logs ${JOB_ID} --region=us-east1
```

### View in Console

```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=psa-scan-scraping
```

---

## Step 5: Access Outputs

### Checkpoints

Training saves checkpoints to GCS during training:

```bash
# List all checkpoints
gsutil ls gs://psa-scan-models-us-east1/checkpoints/

# Download best Phase 2 model
gsutil cp gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth ./
```

Checkpoints saved:
- `phase1_best.pth` - Best back-only pretrained model
- `phase2_best.pth` - Best dual-branch fine-tuned model
- `checkpoint_epoch_N.pth` - Periodic checkpoints (every 5 epochs)

### Final Model Export

After training completes, the model is exported to:

```bash
gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/
  ├── psa_dual_branch_state_dict.pth   # Model weights
  └── psa_dual_branch_config.json      # Model configuration
```

Download for deployment:
```bash
gsutil -m cp -r gs://psa-scan-models-us-east1/models/psa_dual_branch_v1/ ./models/
```

---

## Training Configuration

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Training batch size |
| `--phase1_epochs` | 10 | Back-only pretraining epochs |
| `--phase2_epochs` | 30 | Dual-branch fine-tuning epochs |
| `--lr_phase1` | 1e-3 | Phase 1 learning rate |
| `--lr_phase2` | 3e-4 | Phase 2 learning rate |
| `--lambda_fusion` | 0.7 | Back branch weighting (0.7 = 70% back, 30% front) |
| `--alpha_emd` | 0.7 | EMD loss weight |
| `--beta_edge` | 0.05 | Edge damage loss weight |
| `--beta_center` | 0.1 | Centering loss weight |
| `--use_sampler` | True | Use imbalanced sampler (P ∝ 1/freq^0.5) |

### Customize Training

Modify `scripts/submit_training.sh` to adjust hyperparameters:

```bash
--args="\
--splits_path,gs://psa-scan-models-us-east1/data/splits.json,\
--batch_size,64,\                           # Increase batch size
--phase1_epochs,20,\                        # More pretraining
--phase2_epochs,50,\                        # Longer fine-tuning
--lr_phase2,5e-4,\                          # Higher LR
--lambda_fusion,0.8"                        # More weight on back branch
```

---

## Machine Type Options

Current setup: **n1-standard-8 + 1x Tesla T4 GPU**

### Upgrade for Faster Training

```bash
# More powerful GPU
machine-type=n1-highmem-8,\
accelerator-type=NVIDIA_TESLA_V100,\
accelerator-count=1

# Multiple GPUs (requires code changes for DDP)
machine-type=n1-highmem-16,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=4
```

**Estimated Training Time**:
- **T4 GPU**: ~6-8 hours (Phase 1 + Phase 2)
- **V100 GPU**: ~3-4 hours
- **A100 GPU**: ~2-3 hours

---

## Troubleshooting

### Common Issues

**1. "Image not found in Artifact Registry"**
```bash
# Re-run build script
./scripts/build_and_push.sh
```

**2. "splits.json not found"**
```bash
# Upload splits
gsutil cp splits.json gs://psa-scan-models-us-east1/data/
```

**3. "Out of memory (OOM)"**
```bash
# Reduce batch size in submit script
--batch_size,16
```

**4. "Permission denied"**
```bash
# Grant Vertex AI permissions
gcloud projects add-iam-policy-binding psa-scan-scraping \
  --member="serviceAccount:$(gcloud config get-value account)" \
  --role="roles/aiplatform.user"
```

### View Error Logs

```bash
# Get latest job
JOB_ID=$(gcloud ai custom-jobs list \
  --region=us-east1 \
  --filter="displayName:psa-grade-train" \
  --format="value(name)" \
  --sort-by=~createTime \
  --limit=1)

# Stream logs
gcloud ai custom-jobs stream-logs ${JOB_ID} --region=us-east1
```

---

## Cost Estimation

**Vertex AI Training Costs** (us-east1):
- n1-standard-8: ~$0.38/hour
- Tesla T4 GPU: ~$0.35/hour
- **Total**: ~$0.73/hour

**Estimated Training Cost**:
- 40 epochs (10 Phase 1 + 30 Phase 2) @ 6 hours = **~$4.40**

**Storage Costs**:
- GCS Standard: $0.02/GB/month
- Checkpoints (~5 GB): $0.10/month
- Model export (~500 MB): $0.01/month

---

## Next Steps

After training completes:

1. **Evaluate Model**:
   ```bash
   python src/evaluate.py \
     --checkpoint gs://psa-scan-models-us-east1/checkpoints/phase2_best.pth \
     --splits_path splits.json
   ```

2. **Deploy for Inference**:
   - See `SERVING.md` for deployment instructions
   - Options: Cloud Run, Vertex AI Endpoints, or custom API

3. **Monitor Performance**:
   - Track QWK, MAE, and per-grade accuracy
   - Compare against validation metrics

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                  Vertex AI Custom Job                   │
├─────────────────────────────────────────────────────────┤
│  Container: pytorch/pytorch:2.4-cuda12.1                │
│  Machine: n1-standard-8 (8 vCPUs, 30 GB RAM)            │
│  GPU: 1x NVIDIA Tesla T4 (16 GB VRAM)                   │
│  Region: us-east1 (South Carolina)                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │   Training Code (src/)        │
          │   - Dual-branch architecture  │
          │   - 2-phase curriculum        │
          │   - Imbalanced sampling       │
          │   - GCS checkpointing         │
          └───────────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │  Input: GCS Bucket            │
          │  gs://.../data/splits.json    │
          └───────────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │  Output: GCS Bucket           │
          │  Checkpoints: gs://.../       │
          │  Final Model: gs://.../       │
          └───────────────────────────────┘
```

---

## Questions?

- **Vertex AI Docs**: https://cloud.google.com/vertex-ai/docs/training/custom-training
- **Project GCS Bucket**: `gs://psa-scan-models-us-east1/`
- **Artifact Registry**: `us-east1-docker.pkg.dev/psa-scan-scraping/psa-repo-us-east1/`
