#!/bin/bash
# Local testing WITHOUT GCS - uses downloaded images only

set -e

echo "=========================================="
echo "PSA Grading Model - Local Offline Test"
echo "=========================================="
echo ""

# Check if sample data exists
if [ ! -d "png" ] || [ -z "$(ls -A png 2>/dev/null)" ]; then
    echo "❌ Error: No local images found in ./png/"
    echo ""
    echo "Please download sample data first:"
    echo "  chmod +x scripts/download_sample_data.sh"
    echo "  ./scripts/download_sample_data.sh"
    echo ""
    exit 1
fi

echo "✓ Found local images in ./png/"
echo ""

# Create output directory
mkdir -p ./test_checkpoints

echo "Starting training (OFFLINE mode - no GCS access)..."
echo ""

# Run training WITHOUT --gcs_data_bucket flag (will use local images)
# Use splits_sample.json which contains only the downloaded images
python src/train.py \
  --splits_path splits_sample.json \
  --output_dir ./test_checkpoints \
  --batch_size 4 \
  --num_workers 0 \
  --phase1_epochs 1 \
  --phase2_epochs 1 \
  --lr_phase1 1e-3 \
  --lr_phase2 3e-4 \
  --save_every 1 \
  --no_augment

echo ""
echo "=========================================="
echo "✅ Local test complete!"
echo "=========================================="
echo "Checkpoints saved to: ./test_checkpoints/"
