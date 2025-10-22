output "model_bucket" {
  value = google_storage_bucket.model_bucket.name
}

output "artifact_registry_repo" {
  value = google_artifact_registry_repository.docker_repo.repository_id
}

output "workbench_url" {
  value = "https://console.cloud.google.com/vertex-ai/workbench/instances/${google_workbench_instance.psa_dev.name}/open?project=${var.project_id}"
}
