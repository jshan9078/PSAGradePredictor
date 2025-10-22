import os, subprocess, cv2, numpy as np, matplotlib.pyplot as plt
from src.preprocess import lab_preprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get bucket from environment variable
GCS_DATA_BUCKET = os.getenv('GCS_DATA_BUCKET', 'psa-scan-scraping-dataset')

os.makedirs("testdata", exist_ok=True)
# Copy just one file pair
subprocess.run(f"gsutil cp gs://{GCS_DATA_BUCKET}/png/10/100000380_front.png testdata/", shell=True)
subprocess.run(f"gsutil cp gs://{GCS_DATA_BUCKET}/png/10/100000380_back.png testdata/", shell=True)

img = cv2.imread("testdata/100000380_front.png")[:,:,::-1]
processed = lab_preprocess(img)
print("Processed shape:", processed.shape, "Min:", np.min(processed), "Max:", np.max(processed))

plt.imshow(processed[:,:,0], cmap='gray')
plt.title("CLAHE L-channel")
plt.show()
