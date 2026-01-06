# Docker Deployment Guide

Run the Crustacean Monitoring System in an isolated container on Jetson.

## Prerequisites

1. **NVIDIA Container Runtime** (usually pre-installed on JetPack):
   ```bash
   # Verify it's installed
   docker info | grep -i runtime
   ```

2. **Docker Compose** (if not installed):
   ```bash
   sudo apt-get install docker-compose
   ```

## Quick Start

```bash
# Build the image
docker-compose build

# Run real-time pipeline (with camera)
docker-compose run realtime

# Run offline processing
docker-compose run offline

# Interactive shell for debugging
docker-compose run shell
```

## Camera Access

**CSI Camera**: Requires privileged mode (default in compose file)
```bash
docker-compose run realtime
```

**USB Camera**: Ensure `/dev/video0` exists
```bash
ls -la /dev/video0
docker-compose run realtime
```

## Display Output

For GUI display (e.g., `--display` flag):
```bash
# Allow X11 connections
xhost +local:docker

# Run with display
docker-compose run realtime
```

## Volume Mounts

Output directories are mounted for persistence:
- `./logs` → Container logs
- `./realtime_frames` → Detection results
- `./config` → Configuration (editable without rebuild)

## Running Multiple CV Projects

Each project gets its own container with isolated dependencies:
```bash
# This project
cd crustacean-monitoring && docker-compose run realtime

# Another project
cd ../other-cv-project && docker-compose run inference
```

No conflicts between Python packages, OpenCV versions, etc.

## Building for Different JetPack Versions

```bash
# JetPack 4.6.x (default)
docker-compose build

# JetPack 5.x (Orin)
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r35.2.1-py3 -t crustacean-monitoring .
```

## Troubleshooting

**Camera not found**:
```bash
# Check device exists
ls -la /dev/video*

# For CSI, ensure nvarguscamerasrc works on host first
gst-launch-1.0 nvarguscamerasrc ! fakesink
```

**GPU not accessible**:
```bash
# Verify nvidia runtime
docker run --runtime=nvidia --rm nvidia/cuda:11.0-base nvidia-smi
```

**Permission denied**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then logout/login
```
