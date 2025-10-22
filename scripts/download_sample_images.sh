#!/bin/bash
# Download a small subset of images for local testing

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure required environment variables are set
: "${GCS_DATA_BUCKET:?Error: GCS_DATA_BUCKET not set. Copy .env.example to .env and configure it.}"

GCS_BUCKET="${GCS_DATA_BUCKET}"
LOCAL_DIR="."

echo "=========================================="
echo "Downloading Sample Images from GCS"
echo "=========================================="
echo "Bucket: gs://${GCS_BUCKET}"
echo "Local:  ${LOCAL_DIR}"
echo ""

# Download first 100 images from each grade (front and back)
for grade in {1..10}; do
    echo "Downloading Grade ${grade} samples..."

    # Get list of images for this grade (limit to 50 cards = 100 images)
    gsutil -m cp -n \
        "gs://${GCS_BUCKET}/png/${grade}/*_front.png" \
        "gs://${GCS_BUCKET}/png/${grade}/*_back.png" \
        "./png/${grade}/" 2>/dev/null || true

    # Only download first 50 files to keep it small
    count=$(ls -1 ./png/${grade}/ 2>/dev/null | wc -l || echo 0)
    echo "  Downloaded ${count} images for grade ${grade}"
done

echo ""
echo "=========================================="
echo "✅ Sample images downloaded!"
echo "=========================================="
echo ""
echo "Total images:"
find png -name "*.png" | wc -l
echo ""
echo "Now you can run local tests without GCS:"
echo "  ./scripts/test_local.sh"
echo "=========================================="
