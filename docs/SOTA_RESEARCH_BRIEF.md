# Crustacean Monitoring System - SOTA Research Brief

## Project Overview

Real-time computer vision system for detecting and analyzing crustaceans (crabs and lobsters) in underwater video feeds. Deployed on edge hardware (NVIDIA Jetson Nano 2GB) for marine biology research.

## Current Pipeline Architecture

```
Video Input → Binary Classifier → Frame Selector → Object Detector → Keypoint Detector → Output
```

### Stage 1: Binary Classifier
- **Purpose:** Detect presence/absence of crustacean in frame
- **Input:** Grayscale frames, 320×180 pixels
- **Output:** Binary signal (0/1) per frame
- **Post-processing:** Signal smoothing and rectification
- **Format:** TFLite

### Stage 2: Frame Selector
- **Purpose:** Select highest quality frames from video segments
- **Input:** Binary signal + video frames
- **Output:** Indices of best frames per segment
- **Method:** Two separate quality assessment models (top/bottom views)
- **Format:** TFLite

### Stage 3: Object Detector
- **Purpose:** Localize crustacean and extract ROI
- **Input:** BGR frame, padded to 1280×1280, resized to 640×640
- **Output:** Bounding box, confidence score, class (crab=0, lobster=1)
- **Method:** YOLO-style detection with NMS
- **ROI Output:** Fixed crop 539×561 pixels, grayscale
- **Format:** TFLite

### Stage 4: Keypoint Detector
- **Purpose:** Detect 7 anatomical landmarks on crustacean
- **Input:** Grayscale ROI (539×561)
- **Output:** 14 values (7 keypoints × 2 coordinates)
- **Keypoints:** Crab left/right, left/right eye, carapace end, tail end, last segment
- **Format:** TFLite

## Hardware Constraints

- **Target Device:** NVIDIA Jetson Nano 2GB
- **GPU:** 128-core Maxwell
- **RAM:** 2GB LPDDR4 (shared CPU/GPU)
- **Camera:** CSI (IMX219) or USB webcam
- **Inference:** TFLite Runtime (current), TensorRT preferred for SOTA

## Performance Requirements

- Real-time processing from live camera feed
- Batch processing of recorded video files
- Memory-efficient (models loaded/unloaded sequentially in offline mode)
- Multi-threaded architecture for real-time mode

## Output Data

- Saved frames (JPG) of detected crustaceans
- CSV files with keypoint coordinates
- Metadata (timestamps, confidence scores)
- Hardware metrics (optional monitoring)

## Known Limitations of Current Approach

1. **4 separate models** = complex pipeline, multiple inference calls
2. **Binary classifier** may be redundant if detector confidence is used
3. **Frame selector** adds latency; modern detectors are fast enough to run on more frames
4. **Sequential processing** limits throughput
5. **TFLite** is slower than TensorRT on Jetson

## Research Goals

Find SOTA models/approaches that can:

1. **Simplify the pipeline** — fewer stages, unified models
2. **Improve accuracy** — better detection and keypoint precision
3. **Maintain edge deployment** — must run on Jetson Nano 2GB
4. **Support custom keypoints** — 7 anatomical landmarks specific to crustaceans

## Relevant Search Topics

- Underwater object detection models
- Animal pose estimation / keypoint detection
- Unified detection + pose models (e.g., YOLO-pose, RTMPose)
- Edge-optimized vision models for Jetson
- Marine species detection ML
- Real-time pose estimation on embedded devices
- Video-based animal tracking
- Few-shot / fine-tuning for custom keypoints

## Model Format Requirements

Must export to one of:
- TensorRT (preferred for Jetson)
- ONNX (convertible to TensorRT)
- TFLite (current, but slower)

## Example SOTA Candidates to Evaluate

| Task | Candidates |
|------|------------|
| Unified detect+pose | YOLOv8-pose, YOLOv9-pose, YOLO-NAS-pose |
| Object detection | YOLOv8/v9/v10, RT-DETR, NanoDet |
| Keypoint/pose | RTMPose, ViTPose, MoveNet, MediaPipe |
| Video understanding | VideoMAE, TimeSformer (if temporal context helps) |
| Image quality | NIMA, MUSIQ, HyperIQA |

## Success Criteria

- Fewer pipeline stages (ideally 1-2 models)
- Equal or better detection accuracy
- Equal or better keypoint accuracy
- Runs in real-time on Jetson Nano 2GB
- Active community / maintained codebase
- Clear fine-tuning documentation
