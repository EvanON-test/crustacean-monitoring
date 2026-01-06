# Crustacean Monitoring System - Architecture Documentation

This document provides comprehensive visual documentation of the Crustacean Monitoring System architecture, data flows, and component interactions using Mermaid diagrams.

## Table of Contents

1. [System Overview](#system-overview)
2. [ML Pipeline Flow](#ml-pipeline-flow)
3. [Package Structure](#package-structure)
4. [Class Hierarchy](#class-hierarchy)
5. [Real-time Pipeline Architecture](#real-time-pipeline-architecture)
6. [Offline Pipeline Architecture](#offline-pipeline-architecture)
7. [Thread Communication](#thread-communication)
8. [Model Processing Details](#model-processing-details)
9. [Camera System](#camera-system)
10. [Hardware Monitoring](#hardware-monitoring)
11. [Configuration System](#configuration-system)
12. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Overview

High-level overview of the entire Crustacean Monitoring System:

```mermaid
flowchart TB
    subgraph Input["📹 Input Sources"]
        CAM[Live Camera<br/>CSI/USB]
        VID[Video Files<br/>MP4/AVI]
    end

    subgraph Pipeline["🔄 Processing Pipeline"]
        direction TB
        BC[Binary Classifier<br/>Presence Detection]
        FS[Frame Selector<br/>Quality Assessment]
        OD[Object Detector<br/>Localization]
        KD[Keypoint Detector<br/>Anatomical Landmarks]
    end

    subgraph Output["📊 Output"]
        FRAMES[Saved Frames<br/>JPG Images]
        CSV[Keypoint Data<br/>CSV Files]
        METRICS[Hardware Metrics<br/>Performance Data]
    end

    subgraph Modes["⚙️ Operating Modes"]
        RT[Real-time Pipeline<br/>Live Processing]
        OFF[Offline Pipeline<br/>Batch Processing]
        MON[Monitoring Pipeline<br/>With Metrics]
    end

    CAM --> RT
    VID --> OFF
    VID --> MON
    
    RT --> Pipeline
    OFF --> Pipeline
    MON --> Pipeline
    
    Pipeline --> FRAMES
    Pipeline --> CSV
    MON --> METRICS
```

---

## ML Pipeline Flow

The 4-stage machine learning pipeline for crustacean detection:

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Binary Classification"]
        V1[Video Frames] --> BC[Binary Classifier<br/>TFLite Model]
        BC --> SIG[Binary Signal<br/>0/1 per frame]
        SIG --> SM[Smoothing &<br/>Rectification]
    end

    subgraph Stage2["Stage 2: Frame Selection"]
        SM --> FS[Frame Selector<br/>Top + Bottom Models]
        FS --> IDX[Best Frame<br/>Indices]
    end

    subgraph Stage3["Stage 3: Object Detection"]
        IDX --> EXT[Extract<br/>Frames]
        EXT --> OD[Object Detector<br/>YOLO-style]
        OD --> ROI[Cropped ROI<br/>+ Confidence]
    end

    subgraph Stage4["Stage 4: Keypoint Detection"]
        ROI --> KD[Keypoint Detector<br/>7 Landmarks]
        KD --> KP[Keypoint<br/>Coordinates]
    end

    KP --> OUT[Output:<br/>CSV + Images]

    style Stage1 fill:#e1f5fe
    style Stage2 fill:#fff3e0
    style Stage3 fill:#e8f5e9
    style Stage4 fill:#fce4ec
```

---

## Package Structure

Organization of the `crustacean` Python package:

```mermaid
flowchart TB
    subgraph crustacean["📦 crustacean"]
        direction TB
        
        subgraph core["core/"]
            P[pipeline.py<br/>Base Pipeline]
            OP[offline_pipeline.py<br/>Batch Processing]
            RP[realtime_pipeline.py<br/>Live Processing]
        end
        
        subgraph models["models/"]
            BM[base_model.py<br/>Abstract Base]
            BC[binary_classifier.py]
            FS[frame_selector.py]
            OD[object_detector.py]
            KD[keypoint_detector.py]
        end
        
        subgraph camera["camera/"]
            CAM_B[base_camera.py<br/>Abstract Base]
            GST[gstreamer_camera.py<br/>CSI Cameras]
            OCV[opencv_camera.py<br/>USB Cameras]
        end
        
        subgraph threads["threads/"]
            AT[analysis_thread.py<br/>BC + FS]
            DT[detection_thread.py<br/>Object Detection]
            ST[save_thread.py<br/>Save Results]
        end
        
        subgraph monitoring["monitoring/"]
            HW[hardware_detector.py<br/>Platform Detection]
            BM2[base_monitor.py<br/>Abstract Base]
            JM[jetson_monitor.py]
            PM[pi_monitor.py]
            GM[generic_monitor.py]
        end
        
        subgraph utils["utils/"]
            CFG[config.py<br/>Configuration]
            LOG[logging_setup.py<br/>Logging]
            EXC[exceptions.py<br/>Custom Errors]
            PRF[profiling.py<br/>Performance]
        end
    end

    P --> OP
    P --> RP
    BM --> BC
    BM --> FS
    BM --> OD
    BM --> KD
    CAM_B --> GST
    CAM_B --> OCV
    BM2 --> JM
    BM2 --> PM
    BM2 --> GM
```

---

## Class Hierarchy

Inheritance relationships between classes:

```mermaid
classDiagram
    class Pipeline {
        <<abstract>>
        +config: Config
        +profiler: PerformanceProfiler
        +models: Dict
        +run()*
        +load_models()
        +cleanup()
    }
    
    class OfflinePipeline {
        +video_dir: Path
        +run()
        -_process_video()
    }
    
    class RealtimePipeline {
        +display_mode: bool
        +camera: BaseCamera
        +threads: Dict
        +run()
        -_main_loop()
    }
    
    Pipeline <|-- OfflinePipeline
    Pipeline <|-- RealtimePipeline

    class BaseModel {
        <<abstract>>
        +config: Config
        +interpreter: TFLite
        +load()*
        +predict()
        +unload()
        +preprocess()*
        +postprocess()*
    }
    
    class BinaryClassifier {
        +predict(video)
        -_smooth_signal()
    }
    
    class FrameSelector {
        +top_interpreter
        +bottom_interpreter
        +predict(signal, video)
    }
    
    class ObjectDetector {
        +confidence_threshold
        +predict(frame)
        -_crop_roi()
    }
    
    class KeypointDetector {
        +num_keypoints: 7
        +predict(roi_frames)
    }
    
    BaseModel <|-- BinaryClassifier
    BaseModel <|-- FrameSelector
    BaseModel <|-- ObjectDetector
    BaseModel <|-- KeypointDetector

    class BaseCamera {
        <<abstract>>
        +config: Config
        +open()*
        +read()*
        +release()*
    }
    
    class GStreamerCamera {
        +pipeline: str
        +open()
        +read()
    }
    
    class OpenCVCamera {
        +device: str
        +open()
        +read()
    }
    
    BaseCamera <|-- GStreamerCamera
    BaseCamera <|-- OpenCVCamera

    class BaseMonitor {
        <<abstract>>
        +interval: float
        +output_file: str
        +run()
        +stop()
        +collect_metrics()*
    }
    
    class JetsonMonitor {
        +collect_metrics()
    }
    
    class RaspberryPiMonitor {
        +collect_metrics()
    }
    
    class GenericMonitor {
        +collect_metrics()
    }
    
    BaseMonitor <|-- JetsonMonitor
    BaseMonitor <|-- RaspberryPiMonitor
    BaseMonitor <|-- GenericMonitor
```

---

## Real-time Pipeline Architecture

Multi-threaded architecture for live camera processing:

```mermaid
flowchart TB
    subgraph MainThread["🔵 Main Thread"]
        direction TB
        CAM[Camera Capture] --> MD[Motion Detection]
        MD -->|Motion Detected| COLLECT[Collect Frames<br/>30 frames]
        COLLECT --> SUBMIT[Submit to<br/>Analysis Queue]
        
        RESULTS[Check Results<br/>Queue] --> SAVE_SUBMIT[Submit to<br/>ThreadPool]
        
        DISPLAY[Display Frame<br/>with Overlays]
    end

    subgraph AnalysisThread["🟢 Analysis Thread"]
        direction TB
        AQ[Analysis Queue] --> TEMP[Create Temp<br/>Video]
        TEMP --> BC_RUN[Run Binary<br/>Classifier]
        BC_RUN --> FS_RUN[Run Frame<br/>Selector]
        FS_RUN --> BEST[Select Best<br/>Frame]
        BEST --> DQ_PUT[Put to Detection<br/>Queue]
    end

    subgraph DetectionThread["🟡 Detection Thread"]
        direction TB
        DQ[Detection Queue] --> OD_RUN[Run Object<br/>Detector]
        OD_RUN --> RESULT[Create Detection<br/>Result]
        RESULT --> RQ_PUT[Put to Results<br/>Queue]
    end

    subgraph ThreadPool["🟣 ThreadPool Executor"]
        direction TB
        SAVE1[Save Thread 1]
        SAVE2[Save Thread 2]
        KD_RUN[Run Keypoint<br/>Detector]
        WRITE[Write Files<br/>JPG + CSV]
    end

    SUBMIT --> AQ
    DQ_PUT --> DQ
    RQ_PUT --> RESULTS
    SAVE_SUBMIT --> SAVE1
    SAVE_SUBMIT --> SAVE2
    SAVE1 --> KD_RUN
    SAVE2 --> KD_RUN
    KD_RUN --> WRITE

    style MainThread fill:#e3f2fd
    style AnalysisThread fill:#e8f5e9
    style DetectionThread fill:#fff8e1
    style ThreadPool fill:#f3e5f5
```

---

## Offline Pipeline Architecture

Sequential batch processing for video files:

```mermaid
flowchart TB
    subgraph Input["Input"]
        DIR[Video Directory] --> LIST[List Video Files]
        LIST --> FILTER[Filter Completed]
    end

    subgraph Processing["Processing Loop"]
        FILTER --> LOOP{For Each Video}
        
        LOOP --> BC[Stage 1:<br/>Binary Classifier]
        BC --> FS[Stage 2:<br/>Frame Selector]
        FS --> EXTRACT[Extract Selected<br/>Frames]
        EXTRACT --> OD[Stage 3:<br/>Object Detector]
        OD --> KD[Stage 4:<br/>Keypoint Detector]
        KD --> SAVE[Save Results]
        SAVE --> MARK[Mark Completed]
        MARK --> LOOP
    end

    subgraph Output["Output"]
        SAVE --> CSV[Keypoints CSV]
        SAVE --> FRAMES[Extracted Frames]
        MARK --> LOG[CompletedFiles.txt]
    end

    subgraph Memory["Memory Management"]
        BC -->|unload| FREE1[Free Memory]
        FS -->|unload| FREE2[Free Memory]
        OD -->|unload| FREE3[Free Memory]
        KD -->|unload| FREE4[Free Memory]
    end

    style Input fill:#e1f5fe
    style Processing fill:#fff3e0
    style Output fill:#e8f5e9
    style Memory fill:#ffebee
```

---

## Thread Communication

Queue-based communication between threads:

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant AQ as Analysis Queue
    participant AT as Analysis Thread
    participant DQ as Detection Queue
    participant DT as Detection Thread
    participant RQ as Results Queue
    participant TP as ThreadPool

    Main->>Main: Detect Motion
    Main->>Main: Collect 30 Frames
    Main->>AQ: put(frames, start_frame)
    
    AQ->>AT: get()
    AT->>AT: Create Temp Video
    AT->>AT: Run BC → Signal
    AT->>AT: Run FS → Best Index
    AT->>DQ: put(best_frame, frame_num)
    
    DQ->>DT: get()
    DT->>DT: Run OD → ROI, Conf
    DT->>RQ: put(DetectionResult)
    
    RQ->>Main: get()
    Main->>Main: Check Confidence
    Main->>TP: submit(save_detection)
    
    TP->>TP: Run KD → Keypoints
    TP->>TP: Save JPG + CSV
```

---

## Model Processing Details

### Binary Classifier Processing

```mermaid
flowchart LR
    subgraph Input["Input"]
        VID[Video<br/>N frames]
    end

    subgraph Preprocess["Preprocessing"]
        GRAY[Convert to<br/>Grayscale]
        RESIZE[Resize to<br/>320×180]
        BATCH[Batch<br/>Processing]
    end

    subgraph Inference["TFLite Inference"]
        MODEL[Binary Classifier<br/>Model]
        RAW[Raw Predictions<br/>0.0-1.0]
    end

    subgraph Postprocess["Postprocessing"]
        SMOOTH[Rectangle<br/>Smoothing<br/>γ=20]
        RECT[Rectification<br/>θ=0.5]
        SIGNAL[Binary Signal<br/>0/1 per frame]
    end

    VID --> GRAY --> RESIZE --> BATCH --> MODEL --> RAW --> SMOOTH --> RECT --> SIGNAL
```

### Object Detector Processing

```mermaid
flowchart LR
    subgraph Input["Input"]
        FRAME[BGR Frame<br/>1280×720]
    end

    subgraph Preprocess["Preprocessing"]
        PAD[Pad to Square<br/>1280×1280]
        RESIZE[Resize to<br/>640×640]
    end

    subgraph Inference["YOLO-style Detection"]
        MODEL[Object Detector<br/>TFLite]
        BOXES[Bounding Boxes<br/>+ Confidence]
        NMS[Non-Max<br/>Suppression]
    end

    subgraph Postprocess["ROI Extraction"]
        BEST[Best Detection]
        CROP[Fixed Crop<br/>539×561]
        GRAY[Convert to<br/>Grayscale]
    end

    subgraph Output["Output"]
        ROI[ROI Array]
        CONF[Confidence]
        CLS[Class Index<br/>0=Crab, 1=Lobster]
    end

    FRAME --> PAD --> RESIZE --> MODEL --> BOXES --> NMS --> BEST --> CROP --> GRAY --> ROI
    BEST --> CONF
    BEST --> CLS
```

### Keypoint Detection

```mermaid
flowchart LR
    subgraph Input["Input"]
        ROI[Grayscale ROI<br/>539×561]
    end

    subgraph Inference["TFLite Inference"]
        RESHAPE[Reshape to<br/>1×539×561×1]
        MODEL[Keypoint<br/>Detector]
        RAW[Raw Output<br/>14 values]
    end

    subgraph Output["7 Anatomical Keypoints"]
        KP1[Crab Left<br/>x1, y1]
        KP2[Crab Right<br/>x2, y2]
        KP3[Left Eye<br/>x3, y3]
        KP4[Right Eye<br/>x4, y4]
        KP5[Carapace End<br/>x5, y5]
        KP6[Tail End<br/>x6, y6]
        KP7[Last Segment<br/>x7, y7]
    end

    ROI --> RESHAPE --> MODEL --> RAW
    RAW --> KP1
    RAW --> KP2
    RAW --> KP3
    RAW --> KP4
    RAW --> KP5
    RAW --> KP6
    RAW --> KP7
```

---

## Camera System

Camera initialization and frame capture:

```mermaid
flowchart TB
    subgraph Factory["Camera Factory"]
        CFG[Config:<br/>camera.type]
        CFG -->|csi| GST[GStreamerCamera]
        CFG -->|usb| OCV[OpenCVCamera]
    end

    subgraph GStreamer["GStreamer Pipeline (CSI)"]
        direction LR
        SRC[nvarguscamerasrc] --> CONV[nvvidconv]
        CONV --> FLIP[videoflip<br/>rotate-180]
        FLIP --> BGR[BGRx → BGR]
        BGR --> SINK[appsink]
    end

    subgraph OpenCV["OpenCV Capture (USB)"]
        direction LR
        DEV[/dev/video0] --> CAP[VideoCapture]
        CAP --> PROPS[Set Properties<br/>W×H, FPS]
        PROPS --> READ[read()]
    end

    GST --> GStreamer
    OCV --> OpenCV

    subgraph Output["Frame Output"]
        SINK --> FRAME[BGR Frame<br/>numpy array]
        READ --> FRAME
    end

    style Factory fill:#e1f5fe
    style GStreamer fill:#fff3e0
    style OpenCV fill:#e8f5e9
```

---

## Hardware Monitoring

Platform-specific hardware monitoring:

```mermaid
flowchart TB
    subgraph Detection["Platform Detection"]
        CHECK[detect_hardware()]
        CHECK -->|/etc/nv_tegra_release| JETSON[Jetson]
        CHECK -->|/proc/device-tree/model<br/>contains 'raspberry'| PI[Raspberry Pi]
        CHECK -->|default| GENERIC[Generic]
    end

    subgraph Monitors["Monitor Types"]
        JETSON --> JM[JetsonMonitor<br/>jtop integration]
        PI --> PM[PiMonitor<br/>vcgencmd]
        GENERIC --> GM[GenericMonitor<br/>psutil only]
    end

    subgraph Metrics["Collected Metrics"]
        direction TB
        COMMON[Common Metrics<br/>CPU %, RAM %, Timestamp]
        
        JM --> J_METRICS[Jetson Specific<br/>GPU %, GPU Temp<br/>CPU Temp, Power]
        PM --> P_METRICS[Pi Specific<br/>CPU Temp<br/>Throttle Status]
        GM --> G_METRICS[Generic<br/>CPU Temp if available]
    end

    subgraph Output["Output"]
        J_METRICS --> CSV[metrics.csv]
        P_METRICS --> CSV
        G_METRICS --> CSV
        COMMON --> CSV
    end

    style Detection fill:#e1f5fe
    style Monitors fill:#fff3e0
    style Metrics fill:#e8f5e9
```

---

## Configuration System

YAML configuration with environment variable overrides:

```mermaid
flowchart TB
    subgraph Sources["Configuration Sources"]
        YAML[config/default_config.yaml]
        CUSTOM[Custom YAML File]
        ENV[Environment Variables<br/>CRUSTACEAN_*]
    end

    subgraph Loading["Config Loading"]
        YAML --> LOAD[Config.load()]
        CUSTOM --> LOAD
        LOAD --> MERGE[Merge with<br/>Defaults]
        ENV --> OVERRIDE[Override<br/>Values]
        MERGE --> OVERRIDE
    end

    subgraph Sections["Configuration Sections"]
        direction TB
        MODELS[models:<br/>BC, FS, OD, KD paths]
        CAMERA[camera:<br/>type, resolution, fps]
        REALTIME[realtime:<br/>thresholds, intervals]
        OUTPUT[output:<br/>directories]
        LOGGING[logging:<br/>level, format]
        MONITORING[monitoring:<br/>interval, metrics]
    end

    OVERRIDE --> MODELS
    OVERRIDE --> CAMERA
    OVERRIDE --> REALTIME
    OVERRIDE --> OUTPUT
    OVERRIDE --> LOGGING
    OVERRIDE --> MONITORING

    subgraph Access["Accessing Config"]
        GET[config.get<br/>'models.object_detector.confidence_threshold']
        DOT[Dot Notation<br/>Nested Access]
        DEFAULT[Default Values<br/>if key missing]
    end

    MODELS --> GET
    GET --> DOT
    GET --> DEFAULT

    style Sources fill:#e1f5fe
    style Loading fill:#fff3e0
    style Sections fill:#e8f5e9
    style Access fill:#fce4ec
```

---

## Data Flow Diagrams

### Complete Real-time Data Flow

```mermaid
flowchart TB
    subgraph Camera["📹 Camera Input"]
        CAM[Camera] -->|BGR Frame| MAIN
    end

    subgraph MAIN["Main Thread Processing"]
        direction TB
        MOTION{Motion<br/>Detected?}
        MOTION -->|No| SKIP[Skip Frame]
        MOTION -->|Yes| COLLECT[Collect 30 Frames]
        COLLECT --> QUEUE_A[Analysis Queue]
    end

    subgraph ANALYSIS["Analysis Thread"]
        direction TB
        QUEUE_A --> TEMP_VID[Temp Video]
        TEMP_VID --> BC_PROC[Binary Classifier]
        BC_PROC -->|Signal| FS_PROC[Frame Selector]
        FS_PROC -->|Best Index| SELECT[Select Frame]
        SELECT --> QUEUE_D[Detection Queue]
    end

    subgraph DETECTION["Detection Thread"]
        direction TB
        QUEUE_D --> OD_PROC[Object Detector]
        OD_PROC -->|ROI + Conf| RESULT[Detection Result]
        RESULT --> QUEUE_R[Results Queue]
    end

    subgraph SAVE["Save Operations"]
        direction TB
        QUEUE_R --> CHECK{Confidence<br/>>= 0.75?}
        CHECK -->|No| DISCARD[Discard]
        CHECK -->|Yes| KD_PROC[Keypoint Detector]
        KD_PROC --> WRITE_IMG[Save JPG]
        KD_PROC --> WRITE_CSV[Save CSV]
        KD_PROC --> WRITE_META[Save Metadata]
    end

    subgraph OUTPUT["📁 Output Files"]
        WRITE_IMG --> IMG[timestamp_screenshot.jpg]
        WRITE_CSV --> CSV[timestamp_keypoints.csv]
        WRITE_META --> META[timestamp_metadata.txt]
    end

    style Camera fill:#e3f2fd
    style MAIN fill:#e8f5e9
    style ANALYSIS fill:#fff8e1
    style DETECTION fill:#fce4ec
    style SAVE fill:#f3e5f5
    style OUTPUT fill:#e0f2f1
```

### Offline Pipeline Data Flow

```mermaid
flowchart TB
    subgraph Input["📁 Input"]
        DIR[Video Directory]
        DIR --> FILES[video1.mp4<br/>video2.mp4<br/>...]
    end

    subgraph Loop["Processing Loop"]
        FILES --> NEXT{Next<br/>Video?}
        NEXT -->|Yes| LOAD[Load Video]
        NEXT -->|No| DONE[Complete]
        
        LOAD --> BC[Binary Classifier<br/>→ Signal Array]
        BC --> FS[Frame Selector<br/>→ Best Indices]
        FS --> EXTRACT[Extract Frames<br/>to PNG]
        EXTRACT --> OD[Object Detector<br/>→ ROIs]
        OD --> KD[Keypoint Detector<br/>→ Coordinates]
        KD --> SAVE[Save Results]
        SAVE --> MARK[Mark Complete]
        MARK --> NEXT
    end

    subgraph Output["📊 Output"]
        SAVE --> CSV[video1_keypoints.csv]
        SAVE --> FRAMES[Extracted Frames/]
        MARK --> LOG[CompletedFiles.txt]
    end

    style Input fill:#e1f5fe
    style Loop fill:#fff3e0
    style Output fill:#e8f5e9
```

---

## State Diagrams

### Real-time Pipeline States

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> CameraOpen: Initialize Camera
    CameraOpen --> ModelsLoaded: Load Models
    ModelsLoaded --> ThreadsStarted: Start Threads
    ThreadsStarted --> Running: Begin Main Loop
    
    Running --> MotionDetected: Motion > Threshold
    MotionDetected --> Collecting: Start Collection
    Collecting --> Collecting: Add Frame
    Collecting --> Processing: 30 Frames Collected
    Processing --> Running: Submit to Queue
    
    Running --> Cooldown: Detection Made
    Cooldown --> Running: Cooldown Expired
    
    Running --> ShuttingDown: Ctrl+C / 'q'
    ShuttingDown --> ThreadsStopped: Stop Threads
    ThreadsStopped --> CameraReleased: Release Camera
    CameraReleased --> ModelsUnloaded: Unload Models
    ModelsUnloaded --> [*]
```

### Model Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Unloaded
    
    Unloaded --> Loading: load()
    Loading --> Loaded: Success
    Loading --> Error: Failure
    
    Loaded --> Inferring: predict()
    Inferring --> Loaded: Complete
    Inferring --> Error: Failure
    
    Loaded --> Unloading: unload()
    Unloading --> Unloaded: Complete
    
    Error --> Unloaded: cleanup()
    
    note right of Loaded
        interpreter != None
        input_details set
        output_details set
    end note
    
    note right of Unloaded
        interpreter = None
        Memory freed
    end note
```

---

## Deployment Diagram

### Native Deployment

```mermaid
flowchart TB
    subgraph JetsonNano["NVIDIA Jetson Nano 2GB"]
        subgraph Hardware["Hardware"]
            CPU[ARM Cortex-A57<br/>Quad-core]
            GPU[128-core Maxwell]
            RAM[2GB LPDDR4]
            CSI[CSI Camera Port]
        end
        
        subgraph Software["Software Stack"]
            OS[JetPack / Ubuntu]
            PYTHON[Python 3.9]
            TFLITE[TFLite Runtime]
            OPENCV[OpenCV 4.9]
            GSTREAMER[GStreamer]
        end
        
        subgraph Application["Crustacean Monitor"]
            PIPELINE[Pipeline]
            MODELS[TFLite Models]
            MONITOR[Hardware Monitor<br/>jtop]
        end
    end

    subgraph Camera["Camera"]
        IMX219[IMX219<br/>CSI Camera]
    end

    subgraph Storage["Storage"]
        SD[SD Card / SSD]
        VIDEOS[Video Files]
        RESULTS[Detection Results]
    end

    IMX219 -->|CSI| CSI
    CSI --> GSTREAMER
    GSTREAMER --> PIPELINE
    PIPELINE --> MODELS
    GPU --> MODELS
    PIPELINE --> RESULTS
    VIDEOS --> PIPELINE
    MONITOR --> CPU
    MONITOR --> GPU
    MONITOR --> RAM

    style JetsonNano fill:#e8f5e9
    style Hardware fill:#fff3e0
    style Software fill:#e1f5fe
    style Application fill:#fce4ec
```

### Docker Deployment (Recommended)

```mermaid
flowchart TB
    subgraph JetsonNano["NVIDIA Jetson Nano"]
        subgraph Host["Host System"]
            DOCKER[Docker Engine]
            NVIDIA_RT[NVIDIA Container Runtime]
            CAMERA_DEV[/dev/video0]
        end
        
        subgraph Container["Docker Container"]
            subgraph BaseImage["L4T ML Base Image"]
                CUDA[CUDA + cuDNN]
                OPENCV[OpenCV with CUDA]
                PYTHON[Python 3.9]
            end
            
            subgraph App["Crustacean Monitor"]
                PIPELINE[Pipeline]
                MODELS[TFLite Models]
            end
        end
        
        subgraph Volumes["Mounted Volumes"]
            CONFIG[./config]
            LOGS[./logs]
            OUTPUT[./realtime_frames]
            VIDEO[./processing/video]
        end
    end

    NVIDIA_RT --> Container
    CAMERA_DEV --> Container
    CONFIG --> App
    LOGS --> App
    OUTPUT --> App
    VIDEO --> App

    style Host fill:#e1f5fe
    style Container fill:#e8f5e9
    style Volumes fill:#fff3e0
```

---

## Summary

This architecture documentation covers:

| Component | Description |
|-----------|-------------|
| **ML Pipeline** | 4-stage detection: BC → FS → OD → KD |
| **Operating Modes** | Real-time, Offline, Monitoring |
| **Threading** | Multi-threaded real-time with queues |
| **Camera Support** | CSI (GStreamer) and USB (OpenCV) |
| **Hardware Monitoring** | Jetson, Raspberry Pi, Generic |
| **Configuration** | YAML-based with env var overrides |

The system is optimized for edge deployment on NVIDIA Jetson Nano 2GB, processing live camera feeds or batch video files to detect and analyze crustaceans (crabs and lobsters) using deep learning models.
