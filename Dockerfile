# Crustacean Monitoring System - Jetson Docker Image
# Base: NVIDIA L4T ML image with CUDA, cuDNN, TensorRT, and OpenCV pre-installed
#
# Build:
#   docker build -t crustacean-monitoring .
#
# Run (with camera and GPU access):
#   docker run --runtime=nvidia --device=/dev/video0 -it crustacean-monitoring

# Use NVIDIA's L4T ML base image for Jetson (includes CUDA OpenCV, TensorFlow, etc.)
# For JetPack 4.6.x (Jetson Nano 2GB)
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r32.7.1-py3
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Initialize Git LFS
RUN git lfs install

# Copy requirements first (for Docker layer caching)
COPY requirements.txt requirements-jetson.txt ./

# Install Python dependencies
# Note: OpenCV is already installed in the base image with CUDA support
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -r requirements-jetson.txt

# Copy the application code
COPY . .

# Install the package in development mode
RUN pip3 install --no-cache-dir -e .

# Create directories for outputs (will be mounted as volumes)
RUN mkdir -p /app/logs /app/realtime_frames /app/processing/extracted_frames

# Default command - can be overridden
CMD ["python3", "-c", "print('Crustacean Monitoring System ready. Use docker-compose to run pipelines.')"]
