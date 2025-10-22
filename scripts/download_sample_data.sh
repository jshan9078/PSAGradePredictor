#!/bin/bash
# Download a small subset of images from GCS for local testing

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure required environment variables are set
: "${GCS_DATA_BUCKET:?Error: GCS_DATA_BUCKET not set. Copy .env.example to .env and configure it.}"

GCS_BUCKET="${GCS_DATA_BUCKET}"
NUM_IMAGES=50  # Download 50 images (100 files: front + back)

echo "=========================================="
echo "Downloading sample data from GCS"
echo "=========================================="
echo "Bucket: gs://${GCS_BUCKET}"
echo "Images: ${NUM_IMAGES} (${NUM_IMAGES}x2 files for front+back)"
echo ""

# Create local directories
mkdir -p png

echo "Fetching image list from splits.json..."

# Extract first N image paths from splits.json and create splits_sample.json
python3 - <<'PYTHON'
import json
import sys

# Load splits
with open('splits.json', 'r') as f:
    splits = json.load(f)

# Get first 50 train, 25 val, 25 test
train_sample = splits['train'][:50]
val_sample = splits['val'][:25]
test_sample = splits.get('test', [])[:25]

# Create new splits file with only sampled data
splits_sample = {
    'train': train_sample,
    'val': val_sample,
    'test': test_sample
}

# Save splits_sample.json
with open('splits_sample.json', 'w') as f:
    json.dump(splits_sample, f, indent=2)

# Collect all file paths for download
all_paths = []
for item in train_sample:
    all_paths.append(item['front'])
    all_paths.append(item['back'])
for item in val_sample:
    all_paths.append(item['front'])
    all_paths.append(item['back'])
for item in test_sample:
    all_paths.append(item['front'])
    all_paths.append(item['back'])

# Write to temp file
with open('/tmp/download_paths.txt', 'w') as f:
    for path in all_paths:
        f.write(path + '\n')

print(f"✅ Created splits_sample.json with {len(train_sample)} train, {len(val_sample)} val, {len(test_sample)} test images")
print(f"Found {len(all_paths)} files to download")
print(f"  Train: {len(train_sample)*2} files ({len(train_sample)} images)")
print(f"  Val:   {len(val_sample)*2} files ({len(val_sample)} images)")
print(f"  Test:  {len(test_sample)*2} files ({len(test_sample)} images)")
PYTHON

echo ""
echo "Downloading files from GCS..."

# Download each file
total_files=$(wc -l < /tmp/download_paths.txt)
current=0

while IFS= read -r path; do
    current=$((current + 1))

    # Create parent directory if needed
    dir=$(dirname "$path")
    mkdir -p "$dir"

    # Download if not exists
    if [ ! -f "$path" ]; then
        echo "[$current/$total_files] Downloading $path..."
        gsutil -q cp "gs://${GCS_BUCKET}/$path" "$path" 2>/dev/null || {
            echo "  ⚠️  Failed to download $path (skipping)"
        }
    else
        echo "[$current/$total_files] ✓ Already exists: $path"
    fi
done < /tmp/download_paths.txt

echo ""
echo "=========================================="
echo "✅ Download complete!"
echo "=========================================="
echo ""
echo "Downloaded images are in ./png/"
echo "Created splits_sample.json with downloaded images only"
echo ""
echo "You can now run local training with:"
echo "  ./scripts/test_local_offline.sh"
echo ""
echo "This will use local images (no GCS access needed)"
echo "=========================================="
