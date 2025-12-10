# Configuration Reference

This document describes all configuration options for the Crustacean Monitoring System.

## Configuration File

The default configuration file is located at `config/default_config.yaml`. You can create custom configuration files and specify them with the `--config` flag.

## Configuration Sections

### Models

Configuration for the four ML models in the pipeline.

#### Binary Classifier

Detects crustacean presence in video frames.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `models.binary_classifier.path` | string | `"processing/binary_classifier/save/DS1_A_200_128.tflite"` | Path to TFLite model file |
| `models.binary_classifier.input_width` | int | `320` | Input image width |
| `models.binary_classifier.input_height` | int | `180` | Input image height |
| `models.binary_classifier.batch_size` | int | `1` | Batch size for inference |
| `models.binary_classifier.smoothing_gamma` | int | `20` | Rectangle smoothing parameter |
| `models.binary_classifier.rectify_theta` | float | `0.5` | Rectification threshold |

#### Frame Selector

Selects highest quality frames from video segments.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `models.frame_selector.top_model_path` | string | `"processing/frame_selector/top_con_norm_bal_mse_1000.tflite"` | Path to top quality model |
| `models.frame_selector.bottom_model_path` | string | `"processing/frame_selector/bottom_con_norm_bal_mse_1000.tflite"` | Path to bottom quality model |
| `models.frame_selector.input_width` | int | `320` | Input image width |
| `models.frame_selector.input_height` | int | `180` | Input image height |

#### Object Detector

Locates and classifies crustaceans in frames.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `models.object_detector.path` | string | `"processing/object_detector/best-expanded.tflite"` | Path to TFLite model file |
| `models.object_detector.input_size` | int | `640` | Input image size (square) |
| `models.object_detector.confidence_threshold` | float | `0.75` | Minimum confidence for detections |
| `models.object_detector.fixed_crop_width` | int | `539` | Fixed ROI crop width |
| `models.object_detector.fixed_crop_height` | int | `561` | Fixed ROI crop height |

#### Keypoint Detector

Identifies anatomical landmarks on detected crustaceans.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `models.keypoint_detector.path` | string | `"processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite"` | Path to TFLite model file |
| `models.keypoint_detector.num_keypoints` | int | `7` | Number of keypoints to detect |

### Camera

Configuration for camera capture.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `camera.type` | string | `"csi"` | Camera type: `"csi"` or `"usb"` |
| `camera.width` | int | `1280` | Capture width in pixels |
| `camera.height` | int | `720` | Capture height in pixels |
| `camera.framerate` | int | `45` | Capture framerate |
| `camera.rotation` | int | `180` | Image rotation (0, 90, 180, 270) |
| `camera.device` | string | `"/dev/video0"` | Device path for USB cameras |

**Camera Types:**
- `csi`: NVIDIA CSI camera via GStreamer (Jetson)
- `usb`: USB webcam via OpenCV

### Real-time Processing

Configuration for real-time pipeline behavior.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `realtime.motion_detection_threshold` | int | `15` | Motion detection sensitivity (percentage of changed pixels) |
| `realtime.detection_cooldown` | int | `3` | Seconds to wait between detections |
| `realtime.frames_to_collect` | int | `30` | Number of frames to collect after motion |
| `realtime.process_interval` | int | `30` | Process every N frames for motion |
| `realtime.max_save_threads` | int | `2` | Maximum concurrent save operations |

**Tuning Tips:**
- Lower `motion_detection_threshold` for more sensitive detection
- Increase `detection_cooldown` to reduce duplicate detections
- Increase `process_interval` to reduce CPU load

### Output

Configuration for output directories and files.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output.detections_dir` | string | `"realtime_frames"` | Directory for saved detections |
| `output.benchmark_dir` | string | `"benchmark"` | Directory for benchmark results |
| `output.extracted_frames_dir` | string | `"processing/extracted_frames"` | Temporary frame storage |
| `output.log_dir` | string | `"logs"` | Directory for log files |
| `output.completed_files` | string | `"./CompletedFiles.txt"` | File tracking completed videos |

### Logging

Configuration for the logging system.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `logging.level` | string | `"INFO"` | Log level: DEBUG, INFO, WARNING, ERROR |
| `logging.console` | bool | `true` | Enable console output |
| `logging.file` | bool | `true` | Enable file logging |
| `logging.max_bytes` | int | `10485760` | Max log file size (10MB) |
| `logging.backup_count` | int | `5` | Number of backup log files |

### Monitoring

Configuration for hardware monitoring.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `monitoring.interval` | float | `2.0` | Seconds between metric collections |
| `monitoring.output_file` | string | `"metrics.csv"` | Default metrics output file |

## Environment Variable Overrides

Any configuration value can be overridden using environment variables with the prefix `CRUSTACEAN_` and underscores replacing dots.

**Format:** `CRUSTACEAN_<SECTION>_<KEY>=value`

**Examples:**
```bash
# Override logging level
export CRUSTACEAN_LOGGING_LEVEL=DEBUG

# Override camera type
export CRUSTACEAN_CAMERA_TYPE=usb

# Override confidence threshold
export CRUSTACEAN_MODELS_OBJECT_DETECTOR_CONFIDENCE_THRESHOLD=0.8

# Override motion detection threshold
export CRUSTACEAN_REALTIME_MOTION_DETECTION_THRESHOLD=20
```

## Example Configurations

### High-Performance (Jetson Nano)

```yaml
models:
  binary_classifier:
    path: "processing/binary_classifier/save/DS1_A_200_128.tflite"
  object_detector:
    confidence_threshold: 0.8

camera:
  type: "csi"
  width: 1280
  height: 720
  framerate: 45

realtime:
  motion_detection_threshold: 15
  detection_cooldown: 3
  frames_to_collect: 30
  process_interval: 30

logging:
  level: "INFO"
```

### Low-Power Mode

```yaml
realtime:
  motion_detection_threshold: 20
  detection_cooldown: 5
  frames_to_collect: 20
  process_interval: 60

logging:
  level: "WARNING"
```

### Debug Mode

```yaml
logging:
  level: "DEBUG"
  console: true
  file: true

realtime:
  process_interval: 10
```

### USB Camera Setup

```yaml
camera:
  type: "usb"
  width: 640
  height: 480
  framerate: 30
  device: "/dev/video0"
```

## Accessing Configuration in Code

```python
from crustacean.utils.config import Config

# Load default config
config = Config.load()

# Load custom config
config = Config.load('config/custom.yaml')

# Access values with dot notation
threshold = config.get('models.object_detector.confidence_threshold')
camera_type = config.get('camera.type', 'usb')  # with default

# Access nested sections
models_config = config.get('models')
```

## Validation

The configuration system validates:
- Required fields are present
- Types are correct
- Paths exist (for model files)
- Values are within valid ranges

Invalid configurations will raise `ConfigurationError` with descriptive messages.
