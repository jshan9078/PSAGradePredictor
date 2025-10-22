# Dockerfile for PSA Grading Training on Vertex AI
# Based on official PyTorch image with CUDA 12.1 support

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY splits.json ./

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden by Vertex AI)
ENTRYPOINT ["python", "src/train.py"]
