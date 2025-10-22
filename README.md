# PSA Card Grading Estimator

A deep learning system for automated PSA grading of collectible cards using dual-branch ResNet architecture trained on Google Cloud Vertex AI.

## Overview

This project uses computer vision to predict PSA (Professional Sports Authenticator) grades (1-10) for collectible cards by analyzing both front and back images. The model achieves **0.76 Quadratic Weighted Kappa (QWK)** on validation data.

### Key Features

- **Dual-branch architecture**: Separate ResNet encoders for front/back card images
- **Advanced preprocessing**: LAB color space with CLAHE, edge detection, and gradient features
- **Production-ready**: Containerized training on Vertex AI with GCS integration
- **Well-documented**: Complete training history with chronological model evolution

## Quick Start

### 1. Environment Setup

```bash
# Clone and configure
git clone <your-repo>
cd psa-estimator

# Set up environment variables
cp .env.example .env
# Edit .env with your GCP credentials

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Download Sample Data (Optional)

```bash
# Download 50 sample images for local testing
./scripts/download_sample_data.sh
```

### 3. Training on Vertex AI

```bash
# Build and push Docker image
./scripts/build_and_push.sh

# Upload training data
./scripts/upload_data.sh

# Submit training job
./scripts/submit_training.sh
```

## Project Structure

```
psa-estimator/
├── src/                    # Source code
│   ├── train.py           # Main training script
│   ├── model.py           # Dual-branch ResNet architecture
│   ├── losses.py          # Custom loss functions (CE + EMD)
│   ├── dataset.py         # PyTorch dataset with GCS support
│   ├── preprocess.py      # LAB/CLAHE/edge preprocessing
│   └── gcs_utils.py       # Google Cloud Storage utilities
├── scripts/               # Deployment scripts
│   ├── build_and_push.sh  # Docker build/push to Artifact Registry
│   ├── upload_data.sh     # Upload data to GCS
│   └── submit_training.sh # Submit Vertex AI training job
├── docs/                  # Detailed documentation
│   ├── deployment.md      # Deployment guide
│   ├── architecture.md    # Model architecture details
│   └── development.md     # Local development guide
├── .env.example           # Environment variable template
├── .gitignore            # Comprehensive security exclusions
├── Dockerfile            # Vertex AI training container
├── requirements.txt      # Python dependencies
├── CHANGELOG.md          # Model training history
└── SETUP.md             # Detailed setup instructions
```

## Model Architecture

**Current Best Configuration:**

- **Front branch**: ResNet-18 (11M params)
- **Back branch**: ResNet-34 (21M params)
- **Total parameters**: ~32M
- **Input size**: 384×384 RGB
- **Preprocessing**: LAB color space, CLAHE, Sobel gradients, Laplacian
- **Loss function**: Weighted Cross-Entropy (0.9) + Earth Mover's Distance (0.7)

See [CHANGELOG.md](CHANGELOG.md) for complete model evolution history.

## Performance

### Current Best Model (Epoch 27)

| Metric | Train | Validation |
|--------|-------|------------|
| **QWK** | 0.87 | **0.76** |
| **Loss** | 0.49 | 3.53 |

**Achievement**: Exceeded the target of 0.7+ validation QWK ✅

### Training Configuration

```bash
Phase 1 (Back-only): 0 epochs (skipped)
Phase 2 (Dual-branch): 50 epochs
Learning rate: 3e-4 (with ReduceLROnPlateau)
Regularization: Dropout 0.25, Weight decay 2e-4
Label smoothing: 0.1
Image size: 384×384
Batch size: 16
Optimizer: AdamW
```

## Documentation

- **[SETUP.md](SETUP.md)** - Detailed environment setup guide
- **[CHANGELOG.md](CHANGELOG.md)** - Complete training history and model evolution
- **[docs/deployment.md](docs/deployment.md)** - Vertex AI deployment guide
- **[docs/architecture.md](docs/architecture.md)** - Model architecture deep dive
- **[docs/development.md](docs/development.md)** - Local development workflow

## Key Technologies

- **ML Framework**: PyTorch 2.4, torchvision
- **Cloud Platform**: Google Cloud Platform (GCP)
  - Vertex AI Custom Training
  - Cloud Storage (GCS)
  - Artifact Registry
- **Computer Vision**: OpenCV, Albumentations
- **Infrastructure**: Docker, Terraform

## Dataset

- **Total images**: 9,824 card pairs (front + back)
- **Grades**: PSA 1-10 (10 classes)
- **Split**: 70% train / 15% validation / 15% test
- **Class imbalance**: Grade 2 is rarest (23 samples), Grade 10 most common (983 samples)
- **Storage**: Google Cloud Storage buckets

### Class Distribution

| Grade | Count | Weight |
|-------|-------|--------|
| 1 | 87 | 10.0× |
| 2 | 23 | 10.0× (capped from 42.67×) |
| 3 | 32 | 10.0× (capped from 30.63×) |
| 4 | 176 | 5.57× |
| 5 | 294 | 3.33× |
| 6 | 547 | 1.79× |
| 7 | 889 | 1.10× |
| 8 | 1,582 | 0.62× |
| 9 | 2,211 | 0.44× |
| 10 | 983 | 1.00× (baseline) |

## Environment Variables

All sensitive configuration is managed via environment variables. Required variables:

```bash
PROJECT_ID          # GCP project ID
REGION             # GCP region (e.g., us-east1)
GCS_BUCKET         # Bucket for models/checkpoints
GCS_DATA_BUCKET    # Bucket for training data
REPO_NAME          # Artifact Registry repository name
```

## Training History Highlights

1. **Preprocessing optimization**: 10-50× speedup by switching from scikit-image to OpenCV
2. **Validation bug fix**: Fixed phase mismatch causing unrealistic val loss (281-648 → 2.7)
3. **Scheduler fix**: Added ReduceLROnPlateau to prevent oscillations
4. **Architecture evolution**: ResNet-34/34 → ResNet-18/34 with balanced regularization
5. **Training strategy**: Eliminated back-only pretraining phase (0 epochs Phase 1)
6. **Regularization tuning**: Found optimal balance (dropout 0.25, weight decay 2e-4)

See [CHANGELOG.md](CHANGELOG.md) for complete chronological history with metrics and lessons learned.

## Acknowledgments

- Google Cloud Vertex AI for training infrastructure
- PyTorch team for the framework
- PSA for providing training data
