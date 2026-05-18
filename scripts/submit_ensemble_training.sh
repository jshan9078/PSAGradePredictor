#!/bin/bash
# Submit ensemble training jobs to Vertex AI
# Trains 5 diverse CORAL models with different seeds and slight configuration variations

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

# Docker image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# GCS paths
SPLITS_PATH="gs://${GCS_BUCKET}/data/splits.json"

# Ensemble configuration: 5 models with diversity
# Model 1: Baseline (like Run 7)
# Model 2: Lower regularization
# Model 3: Higher regularization
# Model 4: Deeper back branch (ResNet-50)
# Model 5: Different lambda fusion weight

MODELS=(
    # Model 1: Baseline (Run 7 config)
    "model1:42:0.25:2e-4:34:0.7"
    # Model 2: Lower dropout (less regularization, might catch different patterns)
    "model2:123:0.20:1.5e-4:34:0.7"
    # Model 3: Higher dropout (more regularization, different generalization)
    "model3:456:0.30:2.5e-4:34:0.7"
    # Model 4: Deeper back branch (more capacity for complex features)
    "model4:789:0.25:2e-4:50:0.7"
    # Model 5: Higher lambda (even more weight on back branch)
    "model5:101:0.25:2e-4:34:0.75"
)

# Parse command line arguments
SUBMIT_ALL=false
MODEL_INDEX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SUBMIT_ALL=true
            shift
            ;;
        --model)
            MODEL_INDEX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all | --model N]"
            echo "  --all: Submit all 5 ensemble models"
            echo "  --model N: Submit only model N (1-5)"
            exit 1
            ;;
    esac
done

# Function to submit a single training job
submit_model() {
    local config=$1

    # Parse configuration
    IFS=':' read -r name seed dropout weight_decay back_depth lambda_fusion <<< "$config"

    # Job configuration
    local job_name="psa-ensemble-${name}-$(date +%Y%m%d-%H%M%S)"
    local checkpoint_dir="gs://${GCS_BUCKET}/ensemble/${name}/checkpoints/"
    local model_export_dir="gs://${GCS_BUCKET}/ensemble/${name}/model/"

    echo ""
    echo "=========================================="
    echo "Submitting: ${name}"
    echo "=========================================="
    echo "Configuration:"
    echo "  Seed:         ${seed}"
    echo "  Dropout:      ${dropout}"
    echo "  Weight decay: ${weight_decay}"
    echo "  Back depth:   ResNet-${back_depth}"
    echo "  Lambda:       ${lambda_fusion}"
    echo "  Checkpoints:  ${checkpoint_dir}"
    echo "  Model export: ${model_export_dir}"
    echo ""

    gcloud ai custom-jobs create \
      --region=${REGION} \
      --display-name=${job_name} \
      --project=${PROJECT_ID} \
      --worker-pool-spec=\
machine-type=${MACHINE_TYPE},\
replica-count=1,\
accelerator-type=${ACCELERATOR_TYPE},\
accelerator-count=${ACCELERATOR_COUNT},\
container-image-uri=${IMAGE_URI} \
      --args=--splits_path,${SPLITS_PATH},--output_dir,/tmp/checkpoints,--gcs_data_bucket,${GCS_DATA_BUCKET},--gcs_checkpoint_dir,${checkpoint_dir},--gcs_model_dir,${model_export_dir},--image_size,384,--batch_size,16,--phase1_epochs,0,--phase2_epochs,20,--lr_phase1,1e-3,--lr_phase2,3e-4,--back_depth,${back_depth},--dropout,${dropout},--weight_decay,${weight_decay},--lambda_fusion,${lambda_fusion},--seed,${seed},--use_sampler,--use_coral

    echo ""
    echo "✅ ${name} submitted!"
    echo ""
}

# Main execution
echo ""
echo "=========================================="
echo "PSA ENSEMBLE TRAINING SUBMISSION"
echo "=========================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Image:    ${IMAGE_URI}"
echo "Machine:  ${MACHINE_TYPE} + ${ACCELERATOR_TYPE}"
echo ""
echo "Ensemble Strategy:"
echo "  5 models with different seeds and configurations"
echo "  Training: 20 epochs each (models typically peak at epochs 6-15)"
echo "  Expected diversity → improved ensemble performance"
echo "  Target: Val QWK 0.84-0.88 (vs 0.8359 single model)"
echo ""

if [ "$SUBMIT_ALL" = true ]; then
    echo "Mode: Submitting ALL 5 models"
    echo ""
    read -p "This will submit 5 training jobs (~90 min each, ~$20 total cost). Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    for config in "${MODELS[@]}"; do
        submit_model "$config"
        # Small delay to avoid API rate limits
        sleep 2
    done

    echo ""
    echo "=========================================="
    echo "✅ ALL 5 ENSEMBLE MODELS SUBMITTED!"
    echo "=========================================="
    echo ""
    echo "Monitor progress:"
    echo "  gcloud ai custom-jobs list --region=${REGION} --filter='displayName:psa-ensemble'"
    echo ""
    echo "Next steps:"
    echo "  1. Wait for all jobs to complete (~90 minutes)"
    echo "  2. Run ensemble evaluation:"
    echo "     python src/evaluate_ensemble.py --ensemble_dir gs://${GCS_BUCKET}/ensemble/"
    echo ""

elif [ -n "$MODEL_INDEX" ]; then
    if [ "$MODEL_INDEX" -lt 1 ] || [ "$MODEL_INDEX" -gt 5 ]; then
        echo "Error: Model index must be 1-5"
        exit 1
    fi

    # Array is 0-indexed, user input is 1-indexed
    index=$((MODEL_INDEX - 1))
    config="${MODELS[$index]}"

    echo "Mode: Submitting single model (model${MODEL_INDEX})"
    echo ""

    submit_model "$config"

    echo ""
    echo "=========================================="
    echo "✅ MODEL ${MODEL_INDEX} SUBMITTED!"
    echo "=========================================="
    echo ""

else
    echo "Error: Must specify --all or --model N"
    echo ""
    echo "Usage:"
    echo "  $0 --all           # Submit all 5 models"
    echo "  $0 --model 1       # Submit only model 1"
    echo "  $0 --model 2       # Submit only model 2"
    echo "  ... etc"
    echo ""
    echo "Model configurations:"
    for i in "${!MODELS[@]}"; do
        IFS=':' read -r name seed dropout weight_decay back_depth lambda_fusion <<< "${MODELS[$i]}"
        printf "  %d. %s: seed=%s, dropout=%s, back=ResNet-%s, lambda=%s\n" \
            $((i+1)) "$name" "$seed" "$dropout" "$back_depth" "$lambda_fusion"
    done
    echo ""
    exit 1
fi
