# Environment Setup Guide

## Quick Start

1. **Create your environment configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual Google Cloud credentials:**
   ```bash
   # Edit the file with your preferred editor
   nano .env  # or vim, code, etc.
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Environment File Example

Your `.env` should look like this:

```bash
# Google Cloud Project Configuration
PROJECT_ID=psa-scan-scraping
REGION=us-east1

# GCS Bucket Configuration
GCS_BUCKET=psa-scan-models-us-east1
GCS_DATA_BUCKET=psa-scan-scraping-dataset

# Docker Repository
REPO_NAME=psa-repo-us-east1
IMAGE_NAME=psa-trainer
IMAGE_TAG=latest

# Training Configuration
MACHINE_TYPE=n1-standard-8
ACCELERATOR_TYPE=NVIDIA_TESLA_T4
ACCELERATOR_COUNT=1
```

## Verifying Your Setup

Test that your environment variables are loaded correctly:

```bash
# Load environment variables
source .env

# Verify they're set
echo "Project ID: $PROJECT_ID"
echo "GCS Bucket: $GCS_BUCKET"
echo "Data Bucket: $GCS_DATA_BUCKET"
```

## Running Scripts

All scripts now automatically load environment variables from `.env`:

```bash
# Build and push Docker image
./scripts/build_and_push.sh

# Upload training data
./scripts/upload_data.sh

# Submit training job
./scripts/submit_training.sh

# Download sample data for local testing
./scripts/download_sample_data.sh
```

If a required environment variable is missing, the script will fail with a clear error message:

```
Error: PROJECT_ID not set. Copy .env.example to .env and configure it.
```

## Python Scripts

For Python scripts that need environment variables, use `python-dotenv`:

```python
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access variables
project_id = os.getenv('PROJECT_ID')
gcs_bucket = os.getenv('GCS_BUCKET')
```

This pattern is already implemented in [test.py](test.py).

## Security Notes

- **NEVER** commit `.env` to version control (it's already in `.gitignore`)
- **ALWAYS** use `.env.example` as a template for sharing configuration
- **ROTATE** credentials if they're accidentally exposed
- See [SECURITY.md](SECURITY.md) for detailed security guidance

## Troubleshooting

### Script fails with "not set" error

**Problem:** Environment variables aren't loaded

**Solution:**
1. Ensure `.env` exists: `ls -la .env`
2. Check file permissions: `chmod 644 .env`
3. Verify contents: `cat .env`
4. Make sure no quotes around values unless needed

### Python can't find dotenv

**Problem:** `ModuleNotFoundError: No module named 'dotenv'`

**Solution:**
```bash
pip install python-dotenv
# or
pip install -r requirements.txt
```

### GCS access denied

**Problem:** Scripts can't access Google Cloud Storage

**Solution:**
1. Authenticate: `gcloud auth login`
2. Set project: `gcloud config set project YOUR_PROJECT_ID`
3. Verify permissions:
   ```bash
   gsutil ls gs://$GCS_BUCKET
   ```

### Terraform state file conflicts

**Problem:** Git wants to commit `.tfstate` files

**Solution:**
These are already ignored. If you see them:
```bash
git rm --cached terraform/*.tfstate
git rm --cached terraform/*.backup
```

## Next Steps

- Review [SECURITY.md](SECURITY.md) for security best practices
- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions
- Check [README.md](README.md) for project overview
