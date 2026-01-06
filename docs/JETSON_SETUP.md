# Jetson Nano First-Time Setup Guide

Complete step-by-step guide for setting up the Crustacean Monitoring System on a fresh Jetson Nano.

---

## Prerequisites

### Hardware Required

| Item | Specification |
|------|---------------|
| Jetson Nano | 2GB or 4GB model |
| Power Supply | 5V 4A barrel jack (recommended) or Micro-USB |
| MicroSD Card | 64GB+ UHS-1 (or NVMe SSD for better performance) |
| Camera | IMX219 CSI camera or USB webcam |
| Monitor + Keyboard | For initial setup (can use headless after) |
| Ethernet or WiFi | For downloading packages |

### Software Required

- JetPack 4.6.x flashed to SD card
- Internet connection

---

## Step 1: Flash JetPack and Initial Boot

If your Jetson doesn't have JetPack installed:

1. Download [JetPack 4.6.x](https://developer.nvidia.com/embedded/jetpack) SD card image
2. Flash using [balenaEtcher](https://www.balena.io/etcher/)
3. Insert SD card, connect monitor/keyboard, power on
4. Complete the Ubuntu setup wizard (username, password, timezone)

### Verify JetPack Version

```bash
cat /etc/nv_tegra_release
# Should show: R32 (release), REVISION: 7.x
```

---

## Step 2: System Updates and Dependencies

### Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Install Required System Packages

```bash
# Python and development tools
sudo apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    git-lfs

# GStreamer (for CSI camera)
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# OpenCV with CUDA support (critical for performance)
sudo apt-get install -y python3-opencv
```

### Verify OpenCV CUDA Support

```bash
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
# Should output: CUDA devices: 1
```

---

## Step 3: Install Docker (Recommended Deployment)

Docker provides isolated environments, making it easy to run multiple CV projects without conflicts.

### Install Docker

```bash
# Docker should be pre-installed on JetPack, verify:
docker --version

# If not installed:
sudo apt-get install -y docker.io

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Log out and back in for group change to take effect
logout
```

### Install Docker Compose

```bash
# Install pip version (more up-to-date than apt)
sudo pip3 install docker-compose

# Verify
docker-compose --version
```

### Verify NVIDIA Container Runtime

```bash
# Should be pre-configured on JetPack
docker info | grep -i runtime
# Should show: nvidia

# Test GPU access in container
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi
```

---

## Step 4: Clone the Repository

```bash
cd ~

# Clone the repository
git clone https://github.com/EvanON-test/crustacean-monitoring.git
cd crustacean-monitoring

# Initialize Git LFS and pull model files
git lfs install
git lfs pull

# Verify models downloaded (should be MB, not KB)
ls -lh processing/binary_classifier/save/*.tflite
ls -lh processing/object_detector/*.tflite
```

---

## Step 5: Camera Setup

### Option A: CSI Camera (IMX219)

1. Power off the Jetson
2. Connect the ribbon cable to the CSI port (lift the black tab, insert cable with contacts facing the heatsink, push tab down)
3. Power on

```bash
# Test camera
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
    nvvidconv ! xvimagesink

# Press Ctrl+C to stop
```

If you see video output, the camera is working.

### Option B: USB Camera

```bash
# Check device exists
ls /dev/video*
# Should show /dev/video0

# Test camera
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! xvimagesink
```

### Update Configuration for USB Camera

If using USB instead of CSI, edit `config/default_config.yaml`:

```yaml
camera:
  type: "usb"  # Change from "csi" to "usb"
  device: "/dev/video0"
```

---

## Step 6: Build and Run with Docker

### Build the Docker Image

```bash
cd ~/crustacean-monitoring

# Build (takes 10-15 minutes on first run)
docker-compose build
```

### Test Offline Pipeline

Process pre-recorded videos to verify the system works:

```bash
# Run offline processing
docker-compose run offline

# Check outputs
ls -la realtime_frames/
```

### Test Real-Time Pipeline

```bash
# For display output, allow X11 first
xhost +local:docker

# Run with camera and display
docker-compose run realtime

# Press 'q' to quit
```

### Run in Headless Mode (No Monitor)

```bash
# SSH into Jetson from another machine
ssh jetson@<jetson-ip>

cd ~/crustacean-monitoring

# Run without display
docker-compose run --rm crustacean-base crustacean-realtime

# Or modify docker-compose.yml to remove --display flag
```

---

## Step 7: Verify Everything Works

### Quick Health Check

```bash
# Enter container shell
docker-compose run shell

# Inside container:

# Check all imports work
python3 -c "from crustacean.core import OfflinePipeline, RealtimePipeline; print('OK')"

# Check models load
python3 -c "
from crustacean.utils.config import Config
from crustacean.models import BinaryClassifier
config = Config.load()
bc = BinaryClassifier(config)
bc.load()
print('Model loaded successfully')
bc.unload()
"

# Exit container
exit
```

### Run Test Suite

```bash
docker-compose run shell

# Inside container:
python -m pytest tests/ -v --timeout=300

exit
```

---

## Step 8: Performance Optimization

### Set Jetson to Max Performance Mode

```bash
# Set power mode to MAXN (maximum performance)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify
sudo nvpmodel -q
```

### Install Jetson Stats (for Monitoring)

```bash
sudo -H pip3 install jetson-stats
sudo systemctl restart jtop.service

# Test
jtop
```

### Add Swap Space (Recommended for 2GB Model)

```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Step 9: Configure for Your Use Case

### Edit Configuration

The config file is mounted as a volume, so changes don't require rebuilding:

```bash
nano ~/crustacean-monitoring/config/default_config.yaml
```

Key settings to adjust:

```yaml
# Camera settings
camera:
  type: "csi"           # or "usb"
  width: 1280
  height: 720
  framerate: 45
  rotation: 180         # Adjust based on camera mounting

# Detection sensitivity
realtime:
  motion_detection_threshold: 15   # Lower = more sensitive
  detection_cooldown: 3            # Seconds between detections
  frames_to_collect: 30            # Frames per detection event

# Model confidence
models:
  object_detector:
    confidence_threshold: 0.75     # Lower = more detections (more false positives)
```

### Output Directories

By default, outputs go to:
- `./realtime_frames/` - Saved detections (images + keypoints)
- `./logs/` - Application logs

These are mounted as volumes and persist outside the container.

---

## Step 10: Running in Production

### Auto-Start on Boot

Create a systemd service:

```bash
sudo nano /etc/systemd/system/crustacean.service
```

```ini
[Unit]
Description=Crustacean Monitoring System
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/crustacean-monitoring
ExecStart=/usr/local/bin/docker-compose run --rm crustacean-base crustacean-realtime
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable crustacean.service
sudo systemctl start crustacean.service

# Check status
sudo systemctl status crustacean.service

# View logs
journalctl -u crustacean.service -f
```

### Remote Access

```bash
# Find Jetson IP
hostname -I

# SSH from another machine
ssh jetson@<ip-address>

# Copy files from Jetson
scp -r jetson@<ip>:~/crustacean-monitoring/realtime_frames ./
```

---

## Troubleshooting

### Camera Not Found

```bash
# CSI camera
sudo systemctl restart nvargus-daemon
gst-launch-1.0 nvarguscamerasrc ! fakesink

# USB camera
ls /dev/video*
sudo fuser -k /dev/video0  # Kill any process using camera
```

### Out of Memory

```bash
# Check memory
free -h

# Add swap (see Step 8)

# Reduce processing load in config:
# - Increase process_interval
# - Reduce frames_to_collect
```

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
logout
# Log back in
```

### Models Not Loading (File Size ~130 bytes)

```bash
cd ~/crustacean-monitoring
git lfs pull
ls -lh processing/**/*.tflite
```

### Slow Performance

```bash
# Ensure max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check thermal throttling
cat /sys/devices/virtual/thermal/thermal_zone*/temp
# If > 80000 (80°C), add cooling
```

### Container Can't Access Camera

```bash
# Ensure privileged mode in docker-compose.yml
# Or run with explicit device:
docker run --device=/dev/video0 ...
```

---

## Quick Reference Commands

```bash
# Build
docker-compose build

# Run pipelines
docker-compose run realtime      # Live camera with display
docker-compose run offline       # Process videos
docker-compose run monitor       # With hardware metrics
docker-compose run shell         # Debug shell

# View logs
docker-compose logs -f

# Stop all containers
docker-compose down

# Check system
jtop                             # Hardware monitor
free -h                          # Memory
df -h                            # Disk space
```

---

## Next Steps

1. **Test with your camera** - Run `docker-compose run realtime` and verify detections
2. **Adjust sensitivity** - Tune `motion_detection_threshold` and `confidence_threshold` in config
3. **Set up auto-start** - Configure systemd service for production
4. **Monitor performance** - Use `jtop` to watch temperatures and resource usage
5. **Review outputs** - Check `realtime_frames/` for detection quality

For more details, see:
- [Configuration Reference](CONFIGURATION.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Docker Guide](DOCKER.md)
- [Testing Guide](TESTING_GUIDE.md)
