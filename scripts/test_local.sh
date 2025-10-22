#!/bin/bash
# Test training locally with minimal configuration

set -e

echo "=========================================="
echo "Local Training Test"
echo "=========================================="
echo "Running 1 epoch with small batch to verify setup..."
echo ""

cd /Users/jonathan/Desktop/psa-estimator

# Activate virtual environment if needed
# source venv/bin/activate

# Run training with minimal config
# Note: Images will be loaded from GCS bucket
python src/train.py \
  --splits_path splits.json \
  --output_dir ./test_checkpoints \
  --gcs_data_bucket psa-scan-scraping-dataset \
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
echo ""
echo "Check outputs:"
echo "  ls -lh test_checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Review logs for any errors"
echo "  2. Verify checkpoints were created"
echo "  3. If successful, proceed to Docker build"
echo "=========================================="
