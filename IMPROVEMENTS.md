# Project Improvement Roadmap

Comprehensive analysis of improvements for the Crustacean Monitoring System, prioritized by impact and categorized by difficulty.

---

## Priority Legend
- 🔴 **HIGH**: Critical for production, maintainability, or significant performance gains
- 🟡 **MEDIUM**: Important but not blocking, improves code quality
- 🟢 **LOW**: Nice-to-have, polish, or future enhancements

## Difficulty Scale
- ⭐ **EASY** (1-4 hours): Straightforward changes, minimal refactoring
- ⭐⭐ **MEDIUM** (1-3 days): Requires careful refactoring, testing
- ⭐⭐⭐ **HARD** (1-2 weeks): Major architectural changes, extensive testing

---

## 🔴 HIGH PRIORITY

### 1. Consolidate Real-time Pipeline Duplicates
**Priority**: 🔴 HIGH  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2-3 days  
**Impact**: Maintenance, bug fixes, consistency

**Problem:**
- `realtime_pipeline.py` and `realtime_pipeline_demo.py` share ~90% identical code
- Code duplication leads to inconsistent bug fixes and feature additions
- Changes must be manually synced between files

**Solution:**
Create a base `RealtimePipelineBase` class with shared logic:
```python
# New structure:
processing/
  ├── realtime_base.py          # Shared base class
  ├── realtime_pipeline.py      # Headless (inherits base)
  └── realtime_pipeline_demo.py # Display (inherits base)
```

**Benefits:**
- Single source of truth for core logic
- Easier maintenance and testing
- Faster feature development

**Files to modify:**
- `realtime_pipeline.py`
- `realtime_pipeline_demo.py`
- Create new `processing/realtime_base.py`

---

### 2. Replace Print Statements with Logging System
**Priority**: 🔴 HIGH  
**Difficulty**: ⭐ EASY  
**Estimated Time**: 4-6 hours  
**Impact**: Debugging, production deployment, thread safety

**Problem:**
- 175+ print statements across the codebase
- No log levels (DEBUG, INFO, WARNING, ERROR)
- Print statements from multiple threads interleave unpredictably
- No log file persistence for post-mortem analysis
- Already noted in TODO comments

**Solution:**
```python
import logging
from logging.handlers import RotatingFileHandler

# Setup in each module
logger = logging.getLogger(__name__)

# Replace:
print("BC UTIL: Loading Binary Classifier model...")
# With:
logger.info("Loading Binary Classifier model...")

# Setup centralized config
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('crustacean_monitoring.log', maxBytes=10MB, backupCount=5),
            logging.StreamHandler()
        ]
    )
```

**Benefits:**
- Thread-safe logging with timestamps
- Configurable verbosity levels
- Log rotation for long-running processes
- Better debugging in production

**Files to modify:** All `.py` files (systematic replacement)

---

### 3. Centralized Configuration System
**Priority**: 🔴 HIGH  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 1-2 days  
**Impact**: Flexibility, deployment, parameter tuning

**Problem:**
- Hard-coded magic numbers throughout:
  - Frame dimensions: `320x180`, `640x640`, `1280x720`
  - Thresholds: `0.75` confidence, `15%` motion detection
  - Timing: `3s` cooldown, `2s` monitoring interval
  - Paths: `"./processing/..."` everywhere
- No easy way to tune parameters without code changes
- Different configurations for different deployment scenarios

**Solution:**
Create `config.yaml` or `config.json`:
```yaml
# config.yaml
models:
  binary_classifier:
    path: "processing/binary_classifier/save/DS1_A_200_128.tflite"
    input_width: 320
    input_height: 180
  object_detector:
    path: "processing/object_detector/best-expanded.tflite"
    confidence_threshold: 0.75

realtime:
  motion_detection_threshold: 15  # percentage
  detection_cooldown: 3  # seconds
  frames_to_collect: 30
  process_interval: 60

camera:
  width: 1280
  height: 720
  framerate: 45
  rotate: 180

output:
  detections_dir: "realtime_frames"
  benchmark_dir: "benchmark"
  extracted_frames_dir: "processing/extracted_frames"
```

**Files to create:**
- `config.yaml`
- `config_loader.py`

**Files to modify:** All modules that use hard-coded values

---

### 4. Proper Error Handling and Recovery
**Priority**: 🔴 HIGH  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2-3 days  
**Impact**: Robustness, production stability

**Problem:**
- Inconsistent error handling patterns
- Some exceptions caught with generic `Exception as e`
- Threads can silently fail without proper cleanup
- No retry logic for transient failures
- Resource leaks possible on error paths

**Solution:**
```python
# Custom exceptions
class ModelLoadError(Exception): pass
class CameraInitError(Exception): pass
class InferenceError(Exception): pass

# Consistent error handling
try:
    result = od.process_realtime(frame)
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    # Attempt recovery or graceful degradation
except InferenceError as e:
    logger.warning(f"Inference failed, skipping frame: {e}")
    continue  # Skip this frame, continue processing

# Thread error handling
class RobustThread(Thread):
    def run_with_exception_handling(self):
        try:
            self.run()
        except Exception as e:
            logger.exception(f"Thread {self.name} failed: {e}")
            self.cleanup()
```

**Benefits:**
- More reliable long-running processes
- Better error messages for debugging
- Graceful degradation instead of crashes

---

### 5. Remove Code Duplication in Processing Utils
**Priority**: 🔴 HIGH  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2 days  
**Impact**: Maintenance, bug prevention

**Problem:**
- Each `*_util.py` has duplicate `process()` and `process_realtime()` functions
- Only difference is model loading approach
- Bug fixes must be applied to both versions

**Solution:**
```python
# Unified approach in each util
def process(video, use_preloaded=False):
    """
    Process video through model.
    Args:
        video: VideoCapture object
        use_preloaded: If True, use globally preloaded model
    """
    if use_preloaded:
        model = preloaded_interpreter
    else:
        model = _load_model_temporary()
    
    result = _run_inference(model, video)
    
    if not use_preloaded:
        del model
    
    return result
```

**Files to modify:**
- `processing/binary_classifier_util.py`
- `processing/frame_selector_util.py`
- `processing/object_detector_util.py`
- `processing/keypoint_detector_util.py`

---

## 🟡 MEDIUM PRIORITY

### 6. Add Unit and Integration Tests
**Priority**: 🟡 MEDIUM  
**Difficulty**: ⭐⭐⭐ HARD  
**Estimated Time**: 1-2 weeks  
**Impact**: Code confidence, refactoring safety, CI/CD

**Problem:**
- No test suite visible
- Difficult to refactor with confidence
- Manual testing required for every change

**Solution:**
```python
# tests/
tests/
├── __init__.py
├── test_binary_classifier.py
├── test_frame_selector.py
├── test_object_detector.py
├── test_keypoint_detector.py
├── test_pipeline.py
├── test_realtime_pipeline.py
├── fixtures/
│   ├── test_video.mp4
│   └── test_frames/
└── conftest.py
```

Example test:
```python
# tests/test_binary_classifier.py
import pytest
import numpy as np
from processing import binary_classifier_util as bc

def test_binary_classifier_load():
    bc.load_model()
    assert bc.preloaded_interpreter is not None

def test_classification_output_shape(test_video):
    bc.load_model()
    signal = bc.process_realtime(test_video)
    assert signal.shape[0] > 0
    assert all(s in [0, 1] for s in signal)
```

**Tools needed:**
- `pytest`
- `pytest-cov` (coverage)
- Mock camera and models for fast testing

---

### 6.1 Create requirements.txt
**Priority**: 🟡 MEDIUM
**Difficulty**: ⭐ EASY
**Estimated Time**: 30 minutes
**Impact**: Installation, reproducibility

**Problem:**
- Dependencies only documented in README
- No version pinning
- Manual installation error-prone

**Solution:**
```txt
# requirements.txt
tflite-runtime==2.13.0
numpy==2.0.2
opencv-python==4.9.0
Pillow==10.0.0
psutil==7.0.0
PyYAML==6.0  # For config system

# requirements-jetson.txt (additional)
jetson-stats==4.3.2

# requirements-dev.txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
```

---

### 7. Thread Pool Management
**Priority**: 🟡 MEDIUM  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2 days  
**Impact**: Resource management, scalability

**Problem:**
- SaveDetectionThread spawns unlimited threads
- No thread pool or limit on concurrent detections
- Potential resource exhaustion with rapid detections

**Solution:**
```python
from concurrent.futures import ThreadPoolExecutor

class RealtimePipeline:
    def __init__(self):
        # Limit concurrent save operations
        self.save_executor = ThreadPoolExecutor(max_workers=2)
        
    def handle_detection(self, frame, roi, conf, counter):
        # Submit to thread pool instead of spawning
        future = self.save_executor.submit(
            self._save_detection, frame, roi, conf, counter
        )
        # Optional: track futures for cleanup
        self.active_saves.append(future)
```

---

### 8. Parameterize GStreamer Pipeline
**Priority**: 🟡 MEDIUM  
**Difficulty**: ⭐ EASY  
**Estimated Time**: 2 hours  
**Impact**: Camera flexibility, configuration

**Problem:**
- GStreamer pipeline hard-coded
- Different cameras need code changes
- Rotation fixed at 180 degrees

**Solution:**
```python
# In config.yaml
camera:
  type: "csi"  # or "usb", "rtsp"
  width: 1280
  height: 720
  framerate: 45
  rotation: 180  # 0, 90, 180, 270
  flip_method: "rotate-180"  # or "horizontal-flip", etc.

# In code
def build_gstreamer_pipeline(config):
    if config['camera']['type'] == 'csi':
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={config['camera']['width']},"
            f"height={config['camera']['height']},framerate={config['camera']['framerate']}/1 ! "
            f"nvvidconv ! videoflip method={config['camera']['flip_method']} ! "
            f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=2 sync=false"
        )
    elif config['camera']['type'] == 'usb':
        return f"v4l2src device=/dev/video0 ! ..."
```

---

### 9. Add Performance Profiling
**Priority**: 🟡 MEDIUM  
**Difficulty**: ⭐ EASY  
**Estimated Time**: 4 hours  
**Impact**: Optimization insights

**Problem:**
- No detailed timing breakdowns
- Hard to identify bottlenecks
- Performance regressions unnoticed

**Solution:**
```python
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def profiled_section(name):
    """Context manager for profiling code sections"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f"{name} took {elapsed:.3f}s")

# Usage:
with profiled_section("Binary Classification"):
    signal = bc.process_realtime(capture)

# Add --profile flag
if args.profile:
    profiler = cProfile.Profile()
    profiler.enable()
    # ... run pipeline ...
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('pipeline_profile.prof')
```

---

### 10. Improve Monitoring System Modularity
**Priority**: 🟡 MEDIUM  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 1-2 days  
**Impact**: Code organization, reusability

**Problem:**
- Monitoring system tightly coupled to pipeline
- Hardware detection logic mixed with monitoring
- Difficult to add new hardware platforms

**Solution:**
```python
# monitoring/
monitoring/
├── __init__.py
├── base_monitor.py
├── hardware/
│   ├── jetson_monitor.py
│   ├── pi_monitor.py
│   ├── generic_monitor.py
│   └── detector.py  # Auto-detect hardware
└── metrics/
    ├── cpu.py
    ├── gpu.py
    ├── memory.py
    └── temperature.py
```

---

## 🟢 LOW PRIORITY

### 11. Add Docker Support
**Priority**: 🟢 LOW  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2-3 days  
**Impact**: Deployment ease, reproducibility

**Solution:**
```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-base:r32.7.1
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "realtime_pipeline.py"]
```

---

### 12. Web Dashboard for Monitoring
**Priority**: 🟢 LOW  
**Difficulty**: ⭐⭐⭐ HARD  
**Estimated Time**: 2+ weeks  
**Impact**: User experience, remote monitoring

**Features:**
- Real-time metrics visualization
- Detection gallery
- Live camera feed
- System controls (start/stop/configure)

**Stack:** Flask/FastAPI + React + WebSockets

---

### 13. Model Versioning and Registry
**Priority**: 🟢 LOW  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 3-4 days  
**Impact**: Model management, A/B testing

**Solution:**
```yaml
# models.yaml
models:
  binary_classifier:
    active: "v2.0"
    versions:
      v1.0:
        path: "processing/binary_classifier/save/DS1_A_75_128.tflite"
        accuracy: 0.92
      v2.0:
        path: "processing/binary_classifier/save/DS1_A_200_128.tflite"
        accuracy: 0.95
```

---

### 14. CI/CD Pipeline
**Priority**: 🟢 LOW  
**Difficulty**: ⭐⭐ MEDIUM  
**Estimated Time**: 2-3 days  
**Impact**: Code quality, automation

**Setup GitHub Actions:**
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=processing
      - name: Lint
        run: flake8 .
```

---

### 15. Add Species Classification
**Priority**: 🟢 LOW  
**Difficulty**: ⭐⭐⭐ HARD  
**Estimated Time**: Weeks (requires new training)  
**Impact**: Feature enhancement

**Current:** Object detector classifies as Crab (0) or Lobster (1)  
**Enhancement:** Add species-level classification

---

## Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)
**Goal:** Improve code quality and maintainability
1. ✅ Logging system (4-6 hours)
2. ✅ Configuration system (1-2 days)
3. ✅ requirements.txt (30 min)
4. ✅ Error handling improvements (2-3 days)

### Phase 2: Consolidation (1 week)
**Goal:** Remove duplication
5. ✅ Consolidate realtime pipelines (2-3 days)
6. ✅ Unify processing utils (2 days)

### Phase 3: Testing & Quality (1-2 weeks)
**Goal:** Ensure reliability
7. ✅ Unit tests (1-2 weeks)
8. ✅ Integration tests
9. ✅ Performance profiling (4 hours)

### Phase 4: Deployment (1 week)
**Goal:** Production readiness
10. ✅ Thread pool management (2 days)
11. ✅ Docker support (2-3 days)
12. ✅ CI/CD pipeline (2-3 days)

### Phase 5: Enhancements (Ongoing)
**Goal:** New features and improvements
13. Monitoring dashboard
14. Model versioning
15. Additional features

---

## Quick Wins (Start Here)

These can be done quickly for immediate benefit:

1. **Create requirements.txt** (30 min) ⭐
2. **Add logging to one module** (1 hour) ⭐ - Then replicate pattern
3. **Parameterize one hard-coded value** (30 min) ⭐
4. **Add docstrings to undocumented functions** (2 hours) ⭐
5. **Create config.yaml skeleton** (1 hour) ⭐

---

## Summary Statistics

**Total Identified Improvements:** 15

**By Priority:**
- 🔴 HIGH: 5 improvements (Critical for production)
- 🟡 MEDIUM: 5 improvements (Quality of life)
- 🟢 LOW: 5 improvements (Future enhancements)

**By Difficulty:**
- ⭐ EASY: 5 improvements (~10-15 hours total)
- ⭐⭐ MEDIUM: 8 improvements (~2-4 weeks total)
- ⭐⭐⭐ HARD: 2 improvements (~3-4 weeks total)

**Recommended First Sprint (1-2 weeks):**
1. Logging system
2. Configuration system
3. Error handling
4. requirements.txt
5. Begin consolidating realtime pipelines

This would give you the biggest improvement in code maintainability with reasonable effort.

