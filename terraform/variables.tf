variable "project_id" {
  type        = string
  description = "GCP project ID"
  default     = "psa-scan-scraping"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-east1"
}

variable "zone" {
  type        = string
  description = "GCP zone"
  default     = "us-east1-d"
}

variable "model_bucket_name" {
  type        = string
  description = "Bucket for trained models and checkpoints"
  default     = "psa-scan-models-us-east1"
}

variable "artifact_repo_name" {
  type        = string
  description = "Artifact Registry repo for Docker images"
  default     = "psa-repo-us-east1"
}

