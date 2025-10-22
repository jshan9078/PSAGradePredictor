#!/bin/bash
# Upload training data to GCS

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure required environment variables are set
: "${PROJECT_ID:?Error: PROJECT_ID not set. Copy .env.example to .env and configure it.}"
: "${GCS_BUCKET:?Error: GCS_BUCKET not set. Copy .env.example to .env and configure it.}"

echo "=========================================="
echo "Uploading Data to GCS"
echo "=========================================="
echo "Bucket: gs://${GCS_BUCKET}"
echo "=========================================="

# Upload splits.json
echo ""
echo "1. Uploading splits.json..."
if [ -f "splits.json" ]; then
    gsutil cp splits.json gs://${GCS_BUCKET}/data/splits.json
    echo "✅ splits.json uploaded"
else
    echo "❌ splits.json not found! Run 'python src/split_dataset.py' first"
    exit 1
fi

# Optionally upload dataset manifest
echo ""
echo "2. Uploading dataset_manifest.csv (optional)..."
if [ -f "dataset_manifest.csv" ]; then
    gsutil cp dataset_manifest.csv gs://${GCS_BUCKET}/data/dataset_manifest.csv
    echo "✅ dataset_manifest.csv uploaded"
else
    echo "⚠ dataset_manifest.csv not found (optional)"
fi

echo ""
echo "=========================================="
echo "✅ Data upload complete!"
echo "=========================================="
echo ""
echo "Verify uploads:"
echo "  gsutil ls gs://${GCS_BUCKET}/data/"
echo ""
echo "Next steps:"
echo "  1. Build and push Docker image:"
echo "     ./scripts/build_and_push.sh"
echo ""
echo "  2. Submit training job:"
echo "     ./scripts/submit_training.sh"
echo "=========================================="
