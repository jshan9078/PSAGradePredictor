# ============================
# Enable Required APIs
# ============================
resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com"
  ])
  service = each.key
  disable_on_destroy = false
}

# ============================
# Model + Artifact Buckets
# ============================
resource "google_storage_bucket" "model_bucket" {
  name          = var.model_bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# ============================
# Artifact Registry (for training + serving images)
# ============================
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.artifact_repo_name
  description   = "Docker images for PSA grade training and inference"
  format        = "DOCKER"
}

