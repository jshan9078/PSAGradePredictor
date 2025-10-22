#!/bin/bash
# Build and push Docker image to Artifact Registry

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure required environment variables are set
: "${PROJECT_ID:?Error: PROJECT_ID not set. Copy .env.example to .env and configure it.}"
: "${REGION:=us-east1}"
: "${REPO_NAME:?Error: REPO_NAME not set. Copy .env.example to .env and configure it.}"
: "${IMAGE_NAME:=psa-trainer}"
: "${IMAGE_TAG:=latest}"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building PSA Training Docker Image"
echo "=========================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Image:    ${IMAGE_URI}"
echo "=========================================="

# Configure Docker auth for Artifact Registry (must be done before build)
echo ""
echo "Step 1: Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build the image for linux/amd64 platform (required for Vertex AI)
echo ""
echo "Step 2: Building and pushing Docker image for linux/amd64..."
docker buildx build --platform linux/amd64 -t ${IMAGE_URI} --push .

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "=========================================="
echo "Image URI: ${IMAGE_URI}"
echo ""
echo "Next steps:"
echo "  1. Upload splits.json to GCS:"
echo "     ./scripts/upload_data.sh"
echo ""
echo "  2. Submit training job:"
echo "     ./scripts/submit_training.sh"
echo "=========================================="
