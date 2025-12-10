# Migration Guide

This guide helps you migrate from the legacy crustacean monitoring scripts to the new refactored package structure.

## Overview of Changes

The codebase has been refactored from standalone scripts with duplicated code into a proper Python package with:

- Modular architecture with clear separation of concerns
- YAML-based configuration instead of hardcoded values
- Structured logging instead of print statements
- Custom exception hierarchy for better error handling
- Comprehensive test suite
- Performance profiling tools
- Hardware monitoring abstraction

## Script Mapping

| Legacy Script | New Script | Notes |
|---------------|------------|-------|
| `pipeline.py` | `scripts/run_offline.py` | Now uses CLI arguments |
| `realtime_pipeline.py` | `scripts/run_realtime.py` | Headless mode by default |
| `realtime_pipeline_demo.py` | `scripts/run_realtime.py --display` | Combined into single script |
| `monitoring.py` | `scripts/run_monitoring.py` | Improved metrics collection |

## Command Line Changes

### Offline Processing

**Before:**
```bash
sudo python3.9 pipeline.py
sudo python3.9 pipeline.py --data_path /path/to/videos --runs 4
```

**After:**
```bash
python scripts/run_offline.py --video-dir ./processing/video
python scripts/run_offline.py --video-dir /path/to/videos --profile
```

### Real-time Processing

**Before:**
```bash
sudo python3.9 realtime_pipeline.py
sudo python3.9 realtime_pipeline.py --frames_interval 60
sudo python3.9 realtime_pipeline_demo.py  # With display
```

**After:**
```bash
python scripts/run_realtime.py
python scripts/run_realtime.py --display  # With display
python scripts/run_realtime.py --config config/custom.yaml
```

### Monitoring

**Before:**
```bash
sudo python3.9 monitoring.py --data_path /path/to/videos --runs 8
```

**After:**
```bash
python scripts/run_monitoring.py --video-dir /path/to/videos --output metrics.csv
```

## Code Migration

### Using Models

**Before:**
```python
from processing.binary_classifier_util import binary_classifier, rectangle_smooth, rectify

# Process video
signal = binary_classifier(video_path)
smoothed = rectangle_smooth(signal, gamma=20)
rectified = rectify(smoothed, theta=0.5)
```

**After:**
```python
from crustacean.utils.config import Config
from crustacean.models import BinaryClassifier

config = Config.load()
bc = BinaryClassifier(config)

# Load model
bc.load()

# Process video (smoothing and rectification included)
video = cv2.VideoCapture(str(video_path))
signal = bc.predict(video)
video.release()

# Cleanup
bc.unload()

# Or use context manager
with BinaryClassifier(config) as bc:
    signal = bc.predict(video)
```

### Using Object Detector

**Before:**
```python
from processing.object_detector_util import object_detector

roi, confidence, class_idx = object_detector(frame)
```

**After:**
```python
from crustacean.models import ObjectDetector

with ObjectDetector(config) as od:
    roi, confidence, class_idx = od.predict(frame)
```

### Using Frame Selector

**Before:**
```python
from processing.frame_selector_util import frame_selector

top_indices, bottom_indices = frame_selector(signal, video_path)
```

**After:**
```python
from crustacean.models import FrameSelector

with FrameSelector(config) as fs:
    indices = fs.predict(signal, video)
    top_indices, bottom_indices = indices[0], indices[1]
```

### Using Keypoint Detector

**Before:**
```python
from processing.keypoint_detector_util import keypoint_detector

keypoints = keypoint_detector(roi_frames)
```

**After:**
```python
from crustacean.models import KeypointDetector

with KeypointDetector(config) as kd:
    keypoints = kd.predict(roi_frames)
```

### Running Complete Pipeline

**Before:**
```python
# In pipeline.py - lots of inline code
signal = binary_classifier(video_path)
indices = frame_selector(signal, video_path)
# ... extract frames ...
for frame in frames:
    roi, conf, cls = object_detector(frame)
    keypoints = keypoint_detector(roi)
```

**After:**
```python
from crustacean.utils.config import Config
from crustacean.core import OfflinePipeline

config = Config.load()
pipeline = OfflinePipeline(config, video_dir='./videos')
pipeline.run()  # Handles everything
```

### Hardware Monitoring

**Before:**
```python
from jtop import jtop
import psutil

with jtop() as jetson:
    cpu_temp = jetson.stats['Temp CPU']
    gpu_temp = jetson.stats['Temp GPU']
    cpu_percent = psutil.cpu_percent()
```

**After:**
```python
from crustacean.monitoring import create_monitor, detect_hardware

# Auto-detect platform
hardware = detect_hardware()  # 'jetson', 'raspberry_pi', or 'generic'

# Create appropriate monitor
config = Config.load()
monitor = create_monitor(config, 'metrics.csv')
monitor.start()

# ... run pipeline ...

monitor.stop()
monitor.join()
```

## Configuration Migration

### Hardcoded Values to Config

**Before (hardcoded in scripts):**
```python
MOTION_THRESHOLD = 15
COOLDOWN_SECONDS = 3
FRAMES_TO_COLLECT = 30
PROCESS_INTERVAL = 30
```

**After (in config/default_config.yaml):**
```yaml
realtime:
  motion_detection_threshold: 15
  detection_cooldown: 3
  frames_to_collect: 30
  process_interval: 30
```

### Accessing Configuration

**Before:**
```python
# Hardcoded or command-line args
threshold = 0.75
```

**After:**
```python
from crustacean.utils.config import Config

config = Config.load()
threshold = config.get('models.object_detector.confidence_threshold', 0.75)
```

## Logging Migration

**Before:**
```python
print(f"Processing video: {video_path}")
print(f"Detection confidence: {confidence}")
```

**After:**
```python
from crustacean.utils.logging_setup import get_logger

logger = get_logger(__name__)
logger.info(f"Processing video: {video_path}")
logger.debug(f"Detection confidence: {confidence}")
```

## Error Handling Migration

**Before:**
```python
try:
    model = load_model(path)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

**After:**
```python
from crustacean.utils.exceptions import ModelLoadError, CrustaceanError

try:
    model.load()
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    raise
except CrustaceanError as e:
    logger.error(f"Pipeline error: {e}")
    raise
```

## Directory Structure Changes

```
# Old structure
crustacean-monitoring/
├── pipeline.py
├── realtime_pipeline.py
├── realtime_pipeline_demo.py
├── monitoring.py
└── processing/
    ├── binary_classifier_util.py
    ├── frame_selector_util.py
    ├── object_detector_util.py
    └── keypoint_detector_util.py

# New structure
crustacean-monitoring/
├── crustacean/              # Main package
│   ├── core/                # Pipelines
│   ├── models/              # ML models
│   ├── monitoring/          # Hardware monitoring
│   ├── camera/              # Camera interfaces
│   ├── threads/             # Thread management
│   └── utils/               # Utilities
├── config/                  # Configuration files
├── scripts/                 # Entry points
├── tests/                   # Test suite
└── processing/              # Model files (unchanged)
```

## Breaking Changes

1. **Import paths changed**: All imports now from `crustacean.*` package
2. **Configuration required**: Must use Config object instead of hardcoded values
3. **Model interface changed**: Models now use `load()`/`unload()` pattern
4. **Logging changed**: Uses structured logging instead of print
5. **Script arguments changed**: New CLI argument names and structure
6. **Output format**: Keypoint CSV format slightly different

## Backward Compatibility

The legacy scripts (`pipeline.py`, `realtime_pipeline.py`, etc.) are still present but deprecated. They will be removed in a future version.

To use legacy scripts temporarily:
```bash
# Still works but deprecated
python pipeline.py
```

## Testing Your Migration

After migrating, verify everything works:

```bash
# Run test suite
pytest tests/ -v

# Test offline pipeline
python scripts/run_offline.py --video-dir ./processing/video --log-level DEBUG

# Test real-time (if camera available)
python scripts/run_realtime.py --display --log-level DEBUG

# Test monitoring
python scripts/run_monitoring.py --video-dir ./processing/video --output test_metrics.csv
```

## Getting Help

If you encounter issues during migration:

1. Check the [Configuration Reference](CONFIGURATION.md)
2. Enable debug logging: `--log-level DEBUG`
3. Check log files in `logs/crustacean_monitoring.log`
4. Open an issue on GitHub with error details
