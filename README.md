# Crustacean Monitoring System

A real-time computer vision pipeline for detecting and analyzing crustaceans (crabs and lobsters) on edge devices. This system uses a multi-stage deep learning approach optimized for NVIDIA Jetson Nano 2GB, featuring motion detection, binary classification, frame selection, object detection, and keypoint detection.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Offline Pipeline (Video Files)](#offline-pipeline-video-files)
  - [Real-time Pipeline (Camera Feed)](#real-time-pipeline-camera-feed)
  - [Monitoring System](#monitoring-system)
- [Pipeline Components](#pipeline-components)
- [Output](#output)
- [Project Structure](#project-structure)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system processes video streams to detect and analyze crustaceans through a sophisticated four-stage machine learning pipeline. It's designed for deployment on resource-constrained edge devices, with optimizations for real-time performance including multi-threading, model preloading, and efficient frame processing.

**Key Capabilities:**
- Real-time crustacean detection from live camera feeds
- Offline batch processing of video files
- Anatomical keypoint detection (7 points including eyes, carapace, tail segments)
- Hardware performance monitoring (CPU, GPU, RAM, temperature)
- Multi-threaded architecture for efficient processing

---

## System Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME MODE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Camera Feed                                                             │
│      ↓                                                                   │
│  Motion Detection (15% threshold)                                        │
│      ↓                                                                   │
│  Frame Collection (30 frames)                                            │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────┐                  │
│  │ ANALYSIS THREAD                                   │                  │
│  │  Binary Classifier → Frame Selector               │                  │
│  │  (Detects presence) (Selects best quality frame) │                  │
│  └──────────────────────────────────────────────────┘                  │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────┐                  │
│  │ OBJECT DETECTOR THREAD                            │                  │
│  │  (Locates crustacean, crops ROI)                  │                  │
│  └──────────────────────────────────────────────────┘                  │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────┐                  │
│  │ SAVE DETECTION THREAD                             │                  │
│  │  Keypoint Detector → Save Results                 │                  │
│  │  (7 anatomical points) (Image + CSV)              │                  │
│  └──────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE MODE                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Video File                                                              │
│      ↓                                                                   │
│  Binary Classifier (Process all frames)                                  │
│      ↓                                                                   │
│  Frame Selector (Extract best frames from segments)                      │
│      ↓                                                                   │
│  Object Detector (Detect and crop ROI)                                   │
│      ↓                                                                   │
│  Keypoint Detector (Extract anatomical points)                           │
│      ↓                                                                   │
│  Results Output                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Details

1. **Binary Classifier**: MobileNetV3-Small based model (320x180 input) that detects crustacean presence in frames
   - Uses rectangle smoothing and rectification for noise reduction
   - Outputs binary signal indicating presence/absence

2. **Frame Selector**: Dual CNN models (top/bottom quality assessors) that select the highest quality frames
   - Processes continuous segments where crustaceans are detected
   - Returns indices of optimal frames for further analysis

3. **Object Detector**: YOLOv8-based detector (640x640 input) that locates and classifies crustaceans
   - Supports 2 classes: Crab (0) and Lobster (1)
   - Returns bounding box and confidence score
   - Crops to fixed ROI: 539x561 pixels

4. **Keypoint Detector**: CNN model that identifies 7 anatomical landmarks:
   - Crab left edge (x1, y1)
   - Crab right edge (x2, y2)
   - Left eye (x3, y3)
   - Right eye (x4, y4)
   - Carapace end (x5, y5)
   - Tail end (x6, y6)
   - Last segment (x7, y7)

---

## Features

### Processing Modes

- **Offline Pipeline** (`pipeline.py`): Process pre-recorded video files
- **Real-time Headless** (`realtime_pipeline.py`): Camera processing without display
- **Real-time Demo** (`realtime_pipeline_demo.py`): Camera processing with live visualization
- **Monitoring Mode** (`monitoring.py`): Hardware performance tracking during processing

### Technical Features

- **Multi-threading**: Parallel processing for BC+FS, OD, and KD stages
- **Motion Detection**: Intelligent triggering to reduce computational load
- **Model Preloading**: Persistent models in memory for real-time performance
- **Hardware Monitoring**: CPU, GPU, RAM, temperature tracking
- **Cooldown Mechanism**: Prevents duplicate detections (3s default)
- **Frame Sampling**: Configurable processing intervals (default: every 30 frames)

---

## Hardware Requirements

### Primary Target Device
- **NVIDIA Jetson Nano 2GB Developer Edition**
  - JetPack 4.6.6
  - CUDA 10.2.300
  - cuDNN 8.2.1.32
  - TensorRT 8.2.1.8

### Camera
- CSI camera (via nvarguscamerasrc GStreamer pipeline)
- Recommended: 1280x720 @ 45fps

### Storage
- Minimum 8GB for models and results
- Additional space for video processing and detection storage

### Also Compatible With
- Raspberry Pi (with reduced performance, CPU temp monitoring only)
- Any Linux system with compatible libraries (monitoring features may vary)

---

## Software Requirements

### Core Dependencies

```
Python: 3.9.18
TFLite Runtime: 2.13.0
NumPy: 2.0.2
OpenCV: 4.9.0 (with CUDA support recommended)
Pillow: 10.0.0
psutil: 7.0.0
```

### Jetson-Specific (Optional)
```
jetson-stats: 4.3.2  # For hardware metrics on Jetson devices
```

### Additional Requirements
- GStreamer (for camera capture)
- Git LFS (models are stored with LFS)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/EvanON-test/crustacean-monitoring.git
cd crustacean-monitoring
```

### 2. Install Dependencies

**On Jetson Nano:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.9 python3-pip

# Install Python packages
pip3 install tflite-runtime==2.13.0 numpy==2.0.2 opencv-python==4.9.0 \
             Pillow==10.0.0 psutil==7.0.0 jetson-stats==4.3.2
```

**On other systems:**
```bash
pip3 install tflite-runtime numpy opencv-python Pillow psutil
```

### 3. Verify Models

Ensure all TFLite models are present:
```bash
ls processing/binary_classifier/save/*.tflite
ls processing/frame_selector/*.tflite
ls processing/object_detector/*.tflite
ls processing/keypoint_detector/models/*.tflite
```

---

## Usage

### Offline Pipeline (Video Files)

Process pre-recorded videos:

```bash
# Default usage (processes videos in processing/video/)
sudo python3.9 pipeline.py

# Specify custom video directory
sudo python3.9 pipeline.py --data_path /path/to/videos

# Process with multiple runs (for benchmarking)
sudo python3.9 pipeline.py --data_path /path/to/videos --runs 4
```

**Output:**
- Extracted frames saved to `processing/extracted_frames/`
- Keypoint coordinates printed to console
- Completed files logged to `CompletedFiles.txt`

### Real-time Pipeline (Camera Feed)

#### Headless Mode (No Display)

For deployment scenarios without display:

```bash
# Default: process every 30 frames
sudo python3.9 realtime_pipeline.py

# Custom frame interval
sudo python3.9 realtime_pipeline.py --frames_interval 60
```

#### Demo Mode (With Display)

For development and visualization:

```bash
# Default with live display
sudo python3.9 realtime_pipeline_demo.py

# Custom frame interval
sudo python3.9 realtime_pipeline_demo.py --frames_interval 120
```

**Controls:**
- Press `q` to quit
- `Ctrl+C` for emergency stop

**Output:**
- Detections saved to `realtime_frames/YYYY-MM-DD_HH-MM-SS_Detection/`
- Each detection folder contains:
  - `*_screenshot.jpg`: Original frame
  - `*_keypoints.csv`: Detected keypoint coordinates

### Monitoring System

Track hardware performance during processing:

```bash
# Default monitoring
sudo python3.9 monitoring.py

# Custom configuration
sudo python3.9 monitoring.py --data_path /path/to/videos --runs 8
```

**Output:**
- CSV files saved to `benchmark/`
- Metrics logged every 2 seconds:
  - Timestamp
  - Model stage (BC, FS, OD, KD)
  - CPU percentage
  - CPU temperature
  - GPU temperature (Jetson only)
  - RAM percentage
  - Power usage (when available)

---

## Pipeline Components

### File Structure

**Main Scripts:**
- `pipeline.py` - Offline video processing
- `realtime_pipeline.py` - Real-time headless processing
- `realtime_pipeline_demo.py` - Real-time with visualization
- `monitoring.py` - Hardware monitoring system

**Processing Utilities:**
- `processing/binary_classifier_util.py` - Crustacean presence detection
- `processing/frame_selector_util.py` - Quality-based frame selection
- `processing/object_detector_util.py` - Object detection and ROI extraction
- `processing/keypoint_detector_util.py` - Anatomical landmark detection

**Models:**
- Binary Classifier: `processing/binary_classifier/save/DS1_A_200_128.tflite`
- Frame Selectors: 
  - `processing/frame_selector/top_con_norm_bal_mse_1000.tflite`
  - `processing/frame_selector/bottom_con_norm_bal_mse_1000.tflite`
- Object Detector: `processing/object_detector/best-expanded.tflite`
- Keypoint Detector: `processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite`

---

## Output

### Detection Directory Structure

```
realtime_frames/
└── 2024-10-15_14-30-45_Detection/
    ├── 2024-10-15_14-30-45_screenshot.jpg
    └── 2024-10-15_14-30-45_keypoints.csv
```

### Keypoint CSV Format

```csv
crab_left_x1,crab_left_y1,crab_right_x2,crab_right_y2,left_eye_x3,left_eye_y3,right_eye_x4,right_eye_y4,carapace_end_x5,carapace_end_y5,tail_end_x6,tail_end_y6,last_segment_x7,last_segment_y7
123.4,234.5,345.6,456.7,178.9,200.1,312.3,198.7,250.0,300.0,280.0,450.0,290.0,520.0
```

### Benchmark CSV Format

```csv
timestamp,model_stage,cpu_percent,cpu_temp,gpu_percent,gpu_temp,ram_percent,power_used
08-10-2024_13-30-31,Binary Classifier,45.2,55.0,0.0,48.0,62.3,N/A
08-10-2024_13-30-33,Frame Selector,52.1,56.5,0.0,49.2,63.8,N/A
```

---

## Project Structure

```
crustacean-monitoring/
├── pipeline.py                      # Main offline pipeline
├── realtime_pipeline.py            # Real-time headless
├── realtime_pipeline_demo.py       # Real-time with display
├── monitoring.py                   # Hardware monitoring
├── CompletedFiles.txt              # Processed video log
├── processing/
│   ├── binary_classifier/
│   │   ├── save/                   # TFLite models
│   │   ├── bc_tensorflow/          # Training artifacts
│   │   ├── config/                 # Model configuration
│   │   └── model/                  # Architecture definitions
│   ├── frame_selector/
│   │   ├── top_con_norm_bal_mse_1000.tflite
│   │   └── bottom_con_norm_bal_mse_1000.tflite
│   ├── object_detector/
│   │   ├── best-expanded.tflite    # Main detector
│   │   ├── best.tflite             # Alternative
│   │   ├── nms.py                  # Non-max suppression
│   │   └── od_pytorch/             # PyTorch/ONNX versions
│   ├── keypoint_detector/
│   │   └── models/                 # Current and archived models
│   ├── video/                      # Input videos
│   ├── extracted_frames/           # Temporary frame storage
│   ├── binary_classifier_util.py
│   ├── frame_selector_util.py
│   ├── object_detector_util.py
│   └── keypoint_detector_util.py
├── realtime_frames/                # Real-time detection output
├── benchmark/                      # Monitoring results
└── README.md
```

---

## Performance Considerations

### Optimization Strategies

1. **Frame Interval**: Adjust `--frames_interval` to balance detection rate vs. performance
   - Lower values (30): More frequent processing, higher CPU load
   - Higher values (120-240): Reduced load, might miss fast-moving subjects

2. **Motion Detection Threshold**: Current 15% - adjust in code if needed
   - Higher: Fewer triggers, less processing
   - Lower: More sensitive, more processing

3. **Cooldown Period**: 3-second default prevents duplicate detections
   - Adjust based on deployment scenario

4. **Model Preloading**: Real-time modes keep models in memory
   - Faster inference but higher baseline RAM usage

### Expected Performance (Jetson Nano 2GB)

- **Real-time FPS**: ~30-45 fps camera capture
- **Processing Latency**: 
  - Binary Classifier: ~1-2s per 30 frames
  - Frame Selector: ~0.5-1s
  - Object Detector: ~0.3-0.5s per frame
  - Keypoint Detector: ~0.2-0.4s per frame
- **Total per detection**: ~3-5 seconds from motion to saved result

---

## Troubleshooting

### Common Issues

**Camera not opening:**
```bash
# Check GStreamer pipeline
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! xvimagesink

# Verify camera is detected
ls /dev/video0
```

**Model loading errors:**
```bash
# Verify Git LFS models downloaded
git lfs pull

# Check file sizes (should be MB, not KB)
ls -lh processing/*/models/*.tflite
```

**Permission errors:**
```bash
# Run with sudo for hardware access
sudo python3.9 realtime_pipeline.py
```

**Out of memory:**
- Increase frame interval
- Reduce motion detection sensitivity
- Process fewer concurrent threads

**jtop not available:**
```bash
# Install jetson-stats
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jtop.service
```

### Debug Mode

Add verbose logging by modifying the util files or check console output during execution. All threads print detailed status messages.

---

## Contributing

This is an active research project. Future improvements:
- [ ] Consolidate `realtime_pipeline.py` and `realtime_pipeline_demo.py` (code duplication)
- [ ] Implement logging system instead of print statements
- [ ] Add TensorRT support for improved inference speed
- [ ] Support for additional edge devices
- [ ] Web-based monitoring dashboard
- [ ] Automated species classification refinement

---

## License

[Specify your license here]

## Citation

If you use this system in your research, please cite [relevant publications].

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Repository**: https://github.com/EvanON-test/crustacean-monitoring
