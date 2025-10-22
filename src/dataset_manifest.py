import csv
from google.cloud import storage

def build_manifest(bucket_name="psa-scan-scraping-dataset"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    entries = []
    for grade in range(1, 11):
        for blob in bucket.list_blobs(prefix=f"png/{grade}/"):
            if "front" in blob.name or "back" in blob.name:
                cert_id = blob.name.split("/")[-1].split("_")[0]
                side = "front" if "front" in blob.name else "back"
                entries.append((cert_id, side, grade, blob.name))

    with open("dataset_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cert_id", "side", "grade", "path"])
        writer.writerows(entries)

if __name__ == "__main__":
    build_manifest()
