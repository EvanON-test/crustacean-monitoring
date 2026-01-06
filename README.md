# Crustacean Monitoring System

A real-time computer vision pipeline for detecting and analyzing crustaceans (crabs and lobsters) on edge devices. This system uses a multi-stage deep learning approach optimized for NVIDIA Jetson Nano 2GB, featuring motion detection, binary classification, frame selection, object detection, and keypoint detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)

---

## Overview

This system processes video streams to detect and analyze crustaceans through a sophisticated four-stage machine learning pipeline:

1. **Binary Classifier** - Detects crustacean presence in frames
2. **Frame Selector** - Selects highest quality frames from segments
3. **Object Detector** - Locates and classifies crustaceans (crab/lobster)
4. **Keypoint Detector** - Identifies 7 anatomical landmarks

**Key Capabilities:**
- Real-time crustacean detection from live camera feeds
- Offline batch processing of video files
- Hardware performance monitoring (CPU, GPU, RAM, temperature)
- Multi-threaded architecture for efficient processing
- Configurable via YAML files

---

## Features

### Processing Modes

| Mode | Script | Description |
|------|--------|-------------|
| Offline | `scripts/run_offline.py` | Batch process pre-recorded videos |
| Real-time | `scripts/run_realtime.py` | Live camera processing |
| Monitoring | `scripts/run_monitoring.py` | Offline processing with hardware metrics |

### Technical Features

- **Multi-threading**: Parallel processing for analysis, detection, and saving
- **Motion Detection**: Intelligent triggering to reduce computational load
- **Performance Profiling**: Built-in timing measurements for optimization
- **Hardware Monitoring**: Platform-specific metrics (Jetson, Raspberry Pi, generic)
- **Configurable**: YAML-based configuration with environment variable overrides
- **Structured Logging**: Rotating file logs with configurable levels

---

## Installation

### Requirements

- Python 3.9+
- TFLite Runtime 2.13.0+
- OpenCV 4.9.0+ (CUDA-enabled on Jetson)
- NumPy, Pillow, psutil, PyYAML

### Option 1: Docker (Recommended for Jetson)

Docker provides full isolation, making it easy to run multiple CV projects on the same Jetson without dependency conflicts.

```bash
# Clone repository
git clone https://github.com/EvanON-test/crustacean-monitoring.git
cd crustacean-monitoring

# Build the Docker image
docker-compose build

# Run real-time pipeline (with camera)
docker-compose run realtime

# Run offline processing
docker-compose run offline

# Interactive shell for debugging
docker-compose run shell
```

See [docs/DOCKER.md](docs/DOCKER.md) for detailed Docker usage.

### Option 2: Native Installation

#### Standard Installation

```bash
# Clone repository
git clone https://github.com/EvanON-test/crustacean-monitoring.git
cd crustacean-monitoring

# Install dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

#### Jetson Nano Installation

```bash
# Install system dependencies (CUDA OpenCV is critical for performance)
sudo apt-get update
sudo apt-get install python3.9 python3-pip python3-opencv

# Install Python packages
pip install -r requirements.txt
pip install -r requirements-jetson.txt  # Adds jetson-stats
```

#### Editable Installation (Development)

```bash
pip install -e .
```

### Verify Installation

```bash
# Check imports work
python -c "from crustacean.core import OfflinePipeline, RealtimePipeline; print('OK')"

# Run tests
pytest tests/ -v
```

---

## Quick Start

### Using Docker (Recommended)

```bash
# Process video files
docker-compose run offline

# Real-time detection with camera
docker-compose run realtime

# Monitor hardware during processing
docker-compose run monitor
```

### Using Native Installation

```bash
# Process video files
python scripts/run_offline.py --video-dir ./processing/video

# Real-time detection (headless)
python scripts/run_realtime.py

# Real-time with video display
python scripts/run_realtime.py --display

# Monitor hardware during processing
python scripts/run_monitoring.py --video-dir ./processing/video --output metrics.csv
```

---

## Usage

### Offline Pipeline

Process pre-recorded video files through the 4-stage pipeline:

```bash
# Basic usage
python scripts/run_offline.py --video-dir ./videos

# With custom config and profiling
python scripts/run_offline.py \
    --video-dir ./videos \
    --config config/custom.yaml \
    --profile \
    --log-level DEBUG

# Specify output directory
python scripts/run_offline.py \
    --video-dir ./videos \
    --output-dir ./results
```

**Options:**
- `--video-dir, -v` (required): Directory containing video files
- `--config, -c`: Path to YAML config file
- `--log-level, -l`: DEBUG, INFO, WARNING, ERROR
- `--profile, -p`: Enable performance profiling
- `--output-dir, -o`: Override output directory

### Real-time Pipeline

Process live camera feed:

```bash
# Headless mode (no display)
python scripts/run_realtime.py

# With video display and overlays
python scripts/run_realtime.py --display

# Custom camera type
python scripts/run_realtime.py --camera-type usb

# With profiling
python scripts/run_realtime.py --profile --log-level DEBUG
```

**Options:**
- `--config, -c`: Path to YAML config file
- `--display, -d`: Enable video display with overlays
- `--log-level, -l`: DEBUG, INFO, WARNING, ERROR
- `--profile, -p`: Enable performance profiling
- `--output-dir, -o`: Override output directory
- `--camera-type`: Override camera type (csi, usb)

**Controls:**
- Press `q` to quit (display mode)
- `Ctrl+C` for graceful shutdown

### Monitoring Pipeline

Run offline processing with hardware monitoring:

```bash
# Basic monitoring
python scripts/run_monitoring.py --video-dir ./videos

# Custom output and interval
python scripts/run_monitoring.py \
    --video-dir ./videos \
    --output benchmark/metrics.csv \
    --interval 1.0

# With profiling
python scripts/run_monitoring.py \
    --video-dir ./videos \
    --profile
```

**Options:**
- `--video-dir, -v` (required): Directory containing video files
- `--config, -c`: Path to YAML config file
- `--output, -o`: Metrics CSV output file
- `--log-level, -l`: DEBUG, INFO, WARNING, ERROR
- `--profile, -p`: Enable performance profiling
- `--interval`: Monitoring interval in seconds (default: 2.0)

### Programmatic Usage

```python
from crustacean.utils.config import Config
from crustacean.core import OfflinePipeline, RealtimePipeline
from crustacean.utils.profiling import PerformanceProfiler

# Load configuration
config = Config.load('config/default_config.yaml')

# Create profiler (optional)
profiler = PerformanceProfiler("my_pipeline")

# Run offline pipeline
pipeline = OfflinePipeline(config, video_dir='./videos', profiler=profiler)
pipeline.run()

# Or run real-time pipeline
pipeline = RealtimePipeline(config, display_mode=True, profiler=profiler)
pipeline.run()
```

---

## Configuration

Configuration is managed via YAML files. See `config/default_config.yaml` for all options.

### Key Configuration Sections

```yaml
# Model paths and parameters
models:
  binary_classifier:
    path: "processing/binary_classifier/save/DS1_A_200_128.tflite"
    input_width: 320
    input_height: 180
  
  object_detector:
    path: "processing/object_detector/best-expanded.tflite"
    confidence_threshold: 0.75

# Camera settings
camera:
  type: "csi"  # csi, usb
  width: 1280
  height: 720
  framerate: 45

# Real-time processing
realtime:
  motion_detection_threshold: 15
  detection_cooldown: 3
  frames_to_collect: 30
  process_interval: 30

# Output directories
output:
  detections_dir: "realtime_frames"
  extracted_frames_dir: "processing/extracted_frames"

# Logging
logging:
  level: "INFO"
  console: true
  file: true
```

### Environment Variable Overrides

Override any config value with environment variables:

```bash
export CRUSTACEAN_LOGGING_LEVEL=DEBUG
export CRUSTACEAN_CAMERA_TYPE=usb
python scripts/run_realtime.py
```

---

## Docker Deployment

Docker is the recommended deployment method for Jetson, especially when running multiple CV projects.

### Why Docker?

- Full dependency isolation between projects
- Pre-configured CUDA OpenCV in base image
- Easy deployment and reproducibility
- No conflicts with system Python packages

### Quick Start

```bash
# Build once
docker-compose build

# Run services
docker-compose run realtime   # Live camera processing
docker-compose run offline    # Batch video processing
docker-compose run monitor    # With hardware metrics
docker-compose run shell      # Debug shell
```

### Volume Mounts

Output directories are mounted for persistence:
- `./logs` → Container logs
- `./realtime_frames` → Detection results  
- `./config` → Configuration (editable without rebuild)
- `./processing/video` → Input videos

### Camera Access

```bash
# CSI camera (requires privileged mode, enabled by default)
docker-compose run realtime

# USB camera - ensure device exists
ls /dev/video0
docker-compose run realtime
```

### Display Output

```bash
# Allow X11 connections first
xhost +local:docker

# Then run with display
docker-compose run realtime
```

See [docs/DOCKER.md](docs/DOCKER.md) for complete Docker documentation.

---

## Project Structure

```
crustacean-monitoring/
├── crustacean/                    # Main package
│   ├── core/                      # Pipeline implementations
│   │   ├── pipeline.py           # Base Pipeline class
│   │   ├── offline_pipeline.py   # Batch video processing
│   │   └── realtime_pipeline.py  # Live camera processing
│   ├── models/                    # ML model interfaces
│   │   ├── base_model.py         # Abstract base class
│   │   ├── binary_classifier.py
│   │   ├── frame_selector.py
│   │   ├── object_detector.py
│   │   └── keypoint_detector.py
│   ├── monitoring/                # Hardware metrics
│   │   ├── hardware_detector.py  # Platform detection
│   │   ├── base_monitor.py       # Abstract monitor
│   │   ├── jetson_monitor.py     # Jetson-specific
│   │   ├── pi_monitor.py         # Raspberry Pi
│   │   └── generic_monitor.py    # Cross-platform
│   ├── camera/                    # Camera interfaces
│   │   ├── base_camera.py
│   │   ├── gstreamer_camera.py   # CSI cameras
│   │   └── opencv_camera.py      # USB cameras
│   ├── threads/                   # Thread management
│   │   ├── analysis_thread.py    # BC + FS processing
│   │   ├── detection_thread.py   # Object detection
│   │   └── save_thread.py        # Save detections
│   └── utils/                     # Utilities
│       ├── config.py             # Configuration management
│       ├── logging_setup.py      # Logging configuration
│       ├── exceptions.py         # Custom exceptions
│       └── profiling.py          # Performance profiling
├── config/
│   └── default_config.yaml       # Default configuration
├── scripts/                       # Entry point scripts
│   ├── run_offline.py
│   ├── run_realtime.py
│   └── run_monitoring.py
├── tests/                         # Test suite
│   ├── unit/
│   └── integration/
├── processing/                    # Model files
│   ├── binary_classifier/
│   ├── frame_selector/
│   ├── object_detector/
│   └── keypoint_detector/
├── requirements.txt
├── requirements-dev.txt
├── requirements-jetson.txt
├── setup.py
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker services configuration
└── .dockerignore              # Files excluded from Docker build
```

---

## API Reference

### Core Classes

#### `Pipeline` (Abstract Base)
```python
from crustacean.core import Pipeline

# Methods
pipeline.load_models(preload=False)  # Load all models
pipeline.cleanup()                    # Release resources
pipeline.run()                        # Execute pipeline (abstract)
```

#### `OfflinePipeline`
```python
from crustacean.core import OfflinePipeline

pipeline = OfflinePipeline(
    config=config,
    video_dir='./videos',
    profiler=None  # Optional PerformanceProfiler
)
pipeline.run()
```

#### `RealtimePipeline`
```python
from crustacean.core import RealtimePipeline

pipeline = RealtimePipeline(
    config=config,
    display_mode=False,
    profiler=None
)
pipeline.run()
```

### Model Classes

All models extend `BaseModel` with consistent interface:

```python
from crustacean.models import BinaryClassifier, ObjectDetector

model = BinaryClassifier(config, preload=True)
model.load()           # Load model into memory
result = model.predict(input_data)
model.unload()         # Release resources

# Context manager support
with ObjectDetector(config) as od:
    roi, confidence, class_idx = od.predict(frame)
```

### Monitoring

```python
from crustacean.monitoring import create_monitor, detect_hardware

# Auto-detect platform
hardware = detect_hardware()  # 'jetson', 'raspberry_pi', or 'generic'

# Create appropriate monitor
monitor = create_monitor(config, 'metrics.csv')
monitor.start()
# ... run pipeline ...
monitor.stop()
monitor.join()
```

### Profiling

```python
from crustacean.utils.profiling import PerformanceProfiler

profiler = PerformanceProfiler("my_profiler")

with profiler.profile_section("preprocessing"):
    preprocess_data()

with profiler.profile_section("inference"):
    run_model()

profiler.print_summary()
```

---

## Troubleshooting

### Camera Issues

**Camera not opening:**
```bash
# Check GStreamer pipeline (Jetson)
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! xvimagesink

# Check USB camera
ls /dev/video*

# Try USB camera type
python scripts/run_realtime.py --camera-type usb
```

### Model Issues

**Model loading errors:**
```bash
# Verify Git LFS models downloaded
git lfs pull

# Check file sizes (should be MB, not KB)
ls -lh processing/*/models/*.tflite
ls -lh processing/*/*.tflite
```

### Permission Issues

```bash
# Run with sudo for hardware access (Jetson)
sudo python scripts/run_realtime.py
```

### Memory Issues

- Increase `process_interval` in config (process fewer frames)
- Reduce `frames_to_collect` 
- Use offline mode instead of real-time

### Monitoring Issues

**jtop not available (Jetson):**
```bash
sudo -H pip install -U jetson-stats
sudo systemctl restart jtop.service
```

### Logging

Enable debug logging for detailed output:
```bash
python scripts/run_offline.py --video-dir ./videos --log-level DEBUG
```

Check log files in `logs/crustacean_monitoring.log`

---

## Migration Guide

### From Legacy Scripts

The codebase has been refactored from standalone scripts to a proper Python package.

| Old | New |
|-----|-----|
| `pipeline.py` | `scripts/run_offline.py` |
| `realtime_pipeline.py` | `scripts/run_realtime.py` |
| `realtime_pipeline_demo.py` | `scripts/run_realtime.py --display` |
| `monitoring.py` | `scripts/run_monitoring.py` |

### Key Changes

1. **Configuration**: Now uses YAML files instead of hardcoded values
2. **Logging**: Structured logging replaces print statements
3. **Error Handling**: Custom exception hierarchy
4. **Package Structure**: Proper Python package with imports
5. **Testing**: Comprehensive test suite included

### Updating Existing Code

```python
# Old way
from processing.binary_classifier_util import binary_classifier
signal = binary_classifier(video_path)

# New way
from crustacean.models import BinaryClassifier
from crustacean.utils.config import Config

config = Config.load()
bc = BinaryClassifier(config)
bc.load()
signal = bc.predict(video)
bc.unload()
```

---

## License

[Specify your license here]

## Citation

If you use this system in your research, please cite [relevant publications].

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Repository**: https://github.com/EvanON-test/crustacean-monitoring
