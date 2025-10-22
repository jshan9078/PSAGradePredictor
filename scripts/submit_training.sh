#!/bin/bash
# Submit training job to Vertex AI Custom Training

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
: "${MACHINE_TYPE:=n1-standard-8}"
: "${ACCELERATOR_TYPE:=NVIDIA_TESLA_T4}"
: "${ACCELERATOR_COUNT:=1}"
: "${GCS_BUCKET:?Error: GCS_BUCKET not set. Copy .env.example to .env and configure it.}"
: "${GCS_DATA_BUCKET:?Error: GCS_DATA_BUCKET not set. Copy .env.example to .env and configure it.}"

# Job configuration
JOB_NAME="psa-grade-train-$(date +%Y%m%d-%H%M%S)"

# GCS paths
SPLITS_PATH="gs://${GCS_BUCKET}/data/splits.json"
CHECKPOINT_DIR="gs://${GCS_BUCKET}/checkpoints/"
MODEL_EXPORT_DIR="gs://${GCS_BUCKET}/models/psa_dual_branch_v1/"

# Image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Submitting Vertex AI Training Job"
echo "=========================================="
echo "Job Name:    ${JOB_NAME}"
echo "Region:      ${REGION}"
echo "Machine:     ${MACHINE_TYPE}"
echo "GPU:         ${ACCELERATOR_TYPE} x${ACCELERATOR_COUNT}"
echo "Image:       ${IMAGE_URI}"
echo "Splits:      ${SPLITS_PATH}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Model Export: ${MODEL_EXPORT_DIR}"
echo "=========================================="

# Verify splits.json exists in GCS
echo ""
echo "Verifying splits.json exists in GCS..."
if gsutil ls ${SPLITS_PATH} > /dev/null 2>&1; then
    echo "✅ Found ${SPLITS_PATH}"
else
    echo "❌ Error: ${SPLITS_PATH} not found!"
    echo "   Please upload splits.json first:"
    echo "   gsutil cp splits.json gs://${GCS_BUCKET}/data/"
    exit 1
fi

# Submit training job
echo ""
echo "Submitting training job..."

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --project=${PROJECT_ID} \
  --worker-pool-spec=\
machine-type=${MACHINE_TYPE},\
replica-count=1,\
accelerator-type=${ACCELERATOR_TYPE},\
accelerator-count=${ACCELERATOR_COUNT},\
container-image-uri=${IMAGE_URI} \
  --args=--splits_path,${SPLITS_PATH},--output_dir,/tmp/checkpoints,--gcs_data_bucket,${GCS_DATA_BUCKET},--gcs_checkpoint_dir,${CHECKPOINT_DIR},--gcs_model_dir,${MODEL_EXPORT_DIR},--image_size,384,--batch_size,16,--phase1_epochs,0,--phase2_epochs,50,--lr_phase1,1e-3,--lr_phase2,3e-4,--back_depth,34,--dropout,0.25,--weight_decay,2e-4,--use_sampler

echo ""
echo "=========================================="
echo "✅ Training job submitted!"
echo "=========================================="
echo ""
echo "Monitor your job:"
echo "  gcloud ai custom-jobs list --region=${REGION} --filter=displayName:${JOB_NAME}"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region=${REGION} --filter=displayName:${JOB_NAME} --format='value(name)') --region=${REGION}"
echo ""
echo "Checkpoints will be saved to:"
echo "  ${CHECKPOINT_DIR}"
echo ""
echo "Final model will be exported to:"
echo "  ${MODEL_EXPORT_DIR}"
echo "=========================================="
