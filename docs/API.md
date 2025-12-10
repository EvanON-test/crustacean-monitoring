# API Documentation

This document provides detailed API documentation for the Crustacean Monitoring System.

## Package Structure

```
crustacean/
├── core/           # Pipeline implementations
├── models/         # ML model interfaces
├── monitoring/     # Hardware monitoring
├── camera/         # Camera interfaces
├── threads/        # Thread management
└── utils/          # Utilities
```

---

## crustacean.core

### Pipeline (Abstract Base Class)

```python
from crustacean.core import Pipeline
```

Abstract base class for all pipeline implementations.

#### Constructor

```python
Pipeline(config: Config, profiler: Optional[PerformanceProfiler] = None)
```

**Parameters:**
- `config`: Configuration object with pipeline settings
- `profiler`: Optional performance profiler for timing measurements

#### Methods

| Method | Description |
|--------|-------------|
| `run()` | Execute the pipeline (abstract, must be implemented) |
| `load_models(preload=False)` | Load all required models |
| `cleanup()` | Release all model resources |
| `get_model(name)` | Get a model by name ('bc', 'fs', 'od', 'kd') |
| `is_models_loaded()` | Check if all models are loaded |

#### Context Manager

```python
with Pipeline(config) as pipeline:
    pipeline.run()
# cleanup() called automatically
```

---

### OfflinePipeline

```python
from crustacean.core import OfflinePipeline
```

Batch processing pipeline for pre-recorded video files.

#### Constructor

```python
OfflinePipeline(
    config: Config,
    video_dir: str,
    profiler: Optional[PerformanceProfiler] = None
)
```

**Parameters:**
- `config`: Configuration object
- `video_dir`: Directory containing video files to process
- `profiler`: Optional performance profiler

#### Methods

| Method | Description |
|--------|-------------|
| `run()` | Process all videos in the directory |

#### Example

```python
from crustacean.utils.config import Config
from crustacean.core import OfflinePipeline

config = Config.load()
pipeline = OfflinePipeline(config, video_dir='./videos')
pipeline.run()
```

---

### RealtimePipeline

```python
from crustacean.core import RealtimePipeline
```

Live camera processing pipeline with multi-threading.

#### Constructor

```python
RealtimePipeline(
    config: Config,
    display_mode: bool = False,
    profiler: Optional[PerformanceProfiler] = None
)
```

**Parameters:**
- `config`: Configuration object
- `display_mode`: If True, display video with overlays
- `profiler`: Optional performance profiler

#### Methods

| Method | Description |
|--------|-------------|
| `run()` | Start the real-time processing loop |

#### Example

```python
from crustacean.utils.config import Config
from crustacean.core import RealtimePipeline

config = Config.load()
pipeline = RealtimePipeline(config, display_mode=True)
pipeline.run()  # Press 'q' or Ctrl+C to stop
```

---

## crustacean.models

### BaseModel (Abstract Base Class)

```python
from crustacean.models import BaseModel
```

Abstract base class for all ML models.

#### Constructor

```python
BaseModel(config: Config, preload: bool = False)
```

**Parameters:**
- `config`: Configuration object
- `preload`: If True, load model immediately

#### Methods

| Method | Description |
|--------|-------------|
| `load()` | Load the TFLite model into memory |
| `unload()` | Release model resources |
| `predict(input_data)` | Run inference on input data |
| `preprocess(input_data)` | Preprocess input (abstract) |
| `postprocess(output_data)` | Postprocess output (abstract) |
| `is_loaded()` | Check if model is loaded |

#### Context Manager

```python
with BinaryClassifier(config) as bc:
    result = bc.predict(video)
# unload() called automatically
```

---

### BinaryClassifier

```python
from crustacean.models import BinaryClassifier
```

Detects crustacean presence in video frames.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(video)` | `np.ndarray` | Binary signal indicating presence per frame |

#### Example

```python
import cv2
from crustacean.models import BinaryClassifier

video = cv2.VideoCapture('video.mp4')
with BinaryClassifier(config) as bc:
    signal = bc.predict(video)  # Shape: (n_frames,)
video.release()
```

---

### FrameSelector

```python
from crustacean.models import FrameSelector
```

Selects highest quality frames from video segments.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(signal, video)` | `List[List[int]]` | [top_indices, bottom_indices] |

#### Example

```python
from crustacean.models import FrameSelector

with FrameSelector(config) as fs:
    indices = fs.predict(signal, video)
    top_frames = indices[0]
    bottom_frames = indices[1]
```

---

### ObjectDetector

```python
from crustacean.models import ObjectDetector
```

Locates and classifies crustaceans in frames.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(frame)` | `Tuple[np.ndarray, float, int]` | (roi, confidence, class_index) |

**Class indices:** 0 = Crab, 1 = Lobster

#### Example

```python
from crustacean.models import ObjectDetector

with ObjectDetector(config) as od:
    roi, confidence, class_idx = od.predict(frame)
    if confidence > 0.75:
        print(f"Detected {'Crab' if class_idx == 0 else 'Lobster'}")
```

---

### KeypointDetector

```python
from crustacean.models import KeypointDetector
```

Identifies anatomical landmarks on detected crustaceans.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(roi_frames)` | `np.ndarray` | Keypoint coordinates, shape (n_frames, 14) |

**Keypoint format:** [x1, y1, x2, y2, ..., x7, y7]

#### Example

```python
from crustacean.models import KeypointDetector

with KeypointDetector(config) as kd:
    keypoints = kd.predict(roi_frames)
    # keypoints.shape = (n_frames, 14)
```

---

## crustacean.monitoring

### detect_hardware

```python
from crustacean.monitoring import detect_hardware
```

Auto-detect the hardware platform.

#### Returns

`str`: One of `'jetson'`, `'raspberry_pi'`, or `'generic'`

#### Example

```python
hardware = detect_hardware()
print(f"Running on: {hardware}")
```

---

### create_monitor

```python
from crustacean.monitoring import create_monitor
```

Factory function to create the appropriate monitor for the platform.

#### Signature

```python
create_monitor(config: Config, output_file: str = None) -> BaseMonitor
```

**Parameters:**
- `config`: Configuration object
- `output_file`: Path to output CSV file

**Returns:** Platform-specific monitor instance

#### Example

```python
monitor = create_monitor(config, 'metrics.csv')
monitor.start()
# ... run pipeline ...
monitor.stop()
monitor.join()
```

---

### BaseMonitor

```python
from crustacean.monitoring import BaseMonitor
```

Abstract base class for hardware monitors (extends Thread).

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Start the monitoring thread |
| `stop()` | Signal the thread to stop |
| `join(timeout=None)` | Wait for thread to finish |
| `collect_metrics()` | Collect hardware metrics (abstract) |
| `get_common_metrics()` | Get cross-platform metrics |

---

## crustacean.camera

### create_camera

```python
from crustacean.camera import create_camera
```

Factory function to create the appropriate camera.

#### Signature

```python
create_camera(config: Config) -> BaseCamera
```

#### Example

```python
camera = create_camera(config)
camera.open()
frame = camera.read()
camera.release()
```

---

### BaseCamera (Abstract)

```python
from crustacean.camera import BaseCamera
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `open()` | `bool` | Open camera connection |
| `read()` | `Optional[np.ndarray]` | Read next frame |
| `release()` | `None` | Release camera resources |
| `is_opened()` | `bool` | Check if camera is open |

---

## crustacean.utils

### Config

```python
from crustacean.utils import Config
```

Configuration management class.

#### Class Methods

| Method | Description |
|--------|-------------|
| `Config.load(path=None)` | Load config from file or create default |

#### Instance Methods

| Method | Description |
|--------|-------------|
| `get(key, default=None)` | Get value with dot notation |

#### Example

```python
config = Config.load('config/custom.yaml')
threshold = config.get('models.object_detector.confidence_threshold', 0.75)
```

---

### Logging

```python
from crustacean.utils import setup_logging, get_logger
```

#### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(config)` | Configure logging based on config |
| `get_logger(name)` | Get a logger instance |

#### Example

```python
setup_logging(config)
logger = get_logger(__name__)
logger.info("Processing started")
logger.debug("Debug message")
```

---

### PerformanceProfiler

```python
from crustacean.utils import PerformanceProfiler
```

Performance profiler for measuring execution times.

#### Constructor

```python
PerformanceProfiler(name: str = "profiler", enabled: bool = True)
```

#### Methods

| Method | Description |
|--------|-------------|
| `profile_section(name)` | Context manager for timing a section |
| `get_summary()` | Get statistics dictionary |
| `print_summary()` | Print formatted summary |
| `reset()` | Reset all timing data |

#### Example

```python
profiler = PerformanceProfiler("pipeline")

with profiler.profile_section("preprocessing"):
    preprocess_data()

with profiler.profile_section("inference"):
    run_model()

profiler.print_summary()
```

---

### Exceptions

```python
from crustacean.utils.exceptions import (
    CrustaceanError,
    ConfigurationError,
    ModelLoadError,
    ModelNotLoadedError,
    CameraInitError,
    InferenceError,
    ThreadError,
)
```

| Exception | Description |
|-----------|-------------|
| `CrustaceanError` | Base exception for all errors |
| `ConfigurationError` | Configuration file or validation errors |
| `ModelLoadError` | Model loading failures |
| `ModelNotLoadedError` | Inference without loaded model |
| `CameraInitError` | Camera initialization failures |
| `InferenceError` | Model inference failures |
| `ThreadError` | Thread management errors |

---

## crustacean.threads

### AnalysisThread

```python
from crustacean.threads import AnalysisThread
```

Thread for binary classification and frame selection.

#### Constructor

```python
AnalysisThread(
    analysis_queue: Queue,
    detection_queue: Queue,
    bc_model: BinaryClassifier,
    fs_model: FrameSelector,
    config: Config
)
```

---

### DetectionThread

```python
from crustacean.threads import DetectionThread
```

Thread for object detection.

#### Constructor

```python
DetectionThread(
    frame_queue: Queue,
    result_queue: Queue,
    od_model: ObjectDetector,
    config: Config
)
```

---

### DetectionResult

```python
from crustacean.threads import DetectionResult
```

Data class for detection results.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame` | `np.ndarray` | Original frame |
| `roi` | `np.ndarray` | Cropped ROI |
| `confidence` | `float` | Detection confidence |
| `class_index` | `int` | Class (0=crab, 1=lobster) |
| `frame_number` | `int` | Frame number |

---

### save_detection

```python
from crustacean.threads import save_detection
```

Function to save a detection to disk.

#### Signature

```python
save_detection(
    frame: np.ndarray,
    roi: np.ndarray,
    confidence: float,
    frame_number: int,
    config: Config,
    kd_model: Optional[KeypointDetector] = None
) -> Optional[str]
```

**Returns:** Path to saved detection directory, or None on failure
