# Design Document

## Overview

This design document outlines the architecture and implementation strategy for refactoring the Crustacean Monitoring System. The refactor transforms the current codebase from a collection of loosely coupled scripts with significant duplication into a well-structured, maintainable Python package with proper separation of concerns, comprehensive error handling, and production-ready features.

The design follows a phased approach that allows incremental implementation and testing while maintaining a working system throughout the refactor process.

## Architecture

### High-Level Structure

The refactored system will be organized as a proper Python package with the following structure:

```
crustacean_monitoring/
├── crustacean/                    # Main package
│   ├── __init__.py
│   ├── core/                      # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Base pipeline class
│   │   ├── offline_pipeline.py   # Batch video processing
│   │   └── realtime_pipeline.py  # Live camera processing
│   ├── models/                    # ML model interfaces
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base class
│   │   ├── binary_classifier.py
│   │   ├── frame_selector.py
│   │   ├── object_detector.py
│   │   └── keypoint_detector.py
│   ├── monitoring/                # Hardware metrics
│   │   ├── __init__.py
│   │   ├── base_monitor.py
│   │   ├── hardware_detector.py
│   │   ├── jetson_monitor.py
│   │   ├── pi_monitor.py
│   │   └── generic_monitor.py
│   ├── camera/                    # Camera interfaces
│   │   ├── __init__.py
│   │   ├── base_camera.py
│   │   ├── gstreamer_camera.py
│   │   └── opencv_camera.py
│   ├── utils/                     # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── logging_setup.py      # Logging configuration
│   │   ├── exceptions.py         # Custom exceptions
│   │   └── profiling.py          # Performance profiling
│   └── threads/                   # Thread management
│       ├── __init__.py
│       ├── analysis_thread.py
│       ├── detection_thread.py
│       └── save_thread.py
├── config/                        # Configuration files
│   ├── default_config.yaml
│   └── config_schema.json
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── unit/
│   └── integration/
├── scripts/                       # Entry point scripts
│   ├── run_offline.py
│   ├── run_realtime.py
│   └── run_monitoring.py
├── processing/                    # Model files (unchanged)
│   ├── binary_classifier/
│   ├── frame_selector/
│   ├── object_detector/
│   └── keypoint_detector/
├── requirements.txt
├── requirements-dev.txt
├── requirements-jetson.txt
├── setup.py
└── README.md
```

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Configuration and dependencies passed explicitly
3. **Interface Segregation**: Abstract base classes define contracts
4. **Open/Closed**: Extensible without modifying existing code
5. **DRY (Don't Repeat Yourself)**: Eliminate all code duplication


## Components and Interfaces

### 1. Configuration System

**Module**: `crustacean/utils/config.py`

**Purpose**: Centralized configuration management with validation and defaults

**Key Classes**:
```python
class Config:
    """Main configuration class loaded from YAML"""
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load config from file or create default"""
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        
    def validate(self) -> bool:
        """Validate configuration against schema"""
```

**Configuration Schema** (YAML):
```yaml
# Model configurations
models:
  binary_classifier:
    path: "processing/binary_classifier/save/DS1_A_200_128.tflite"
    input_width: 320
    input_height: 180
    batch_size: 1
    smoothing_gamma: 20
    rectify_theta: 0.5
    
  frame_selector:
    top_model_path: "processing/frame_selector/top_con_norm_bal_mse_1000.tflite"
    bottom_model_path: "processing/frame_selector/bottom_con_norm_bal_mse_1000.tflite"
    input_width: 320
    input_height: 180
    
  object_detector:
    path: "processing/object_detector/best-expanded.tflite"
    input_size: 640
    confidence_threshold: 0.75
    fixed_crop_width: 539
    fixed_crop_height: 561
    
  keypoint_detector:
    path: "processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite"
    num_keypoints: 7

# Real-time processing
realtime:
  motion_detection_threshold: 15  # percentage
  detection_cooldown: 3  # seconds
  frames_to_collect: 30
  process_interval: 30  # process every N frames
  max_save_threads: 2

# Camera configuration
camera:
  type: "csi"  # csi, usb, rtsp
  width: 1280
  height: 720
  framerate: 45
  rotation: 180
  device: "/dev/video0"  # for USB cameras

# Output paths
output:
  detections_dir: "realtime_frames"
  benchmark_dir: "benchmark"
  extracted_frames_dir: "processing/extracted_frames"
  log_dir: "logs"

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  console: true
  file: true
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

**Design Decisions**:
- Use YAML for human-readable configuration
- Dot notation for nested access (e.g., `config.get('models.binary_classifier.path')`)
- JSON schema validation to catch errors early
- Environment variable overrides (e.g., `CRUSTACEAN_LOG_LEVEL=DEBUG`)


### 2. Logging System

**Module**: `crustacean/utils/logging_setup.py`

**Purpose**: Structured logging with thread-safe output and rotation

**Key Functions**:
```python
def setup_logging(config: Config) -> None:
    """Configure logging based on config settings"""
    
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
```

**Log Format**:
```
2024-11-12 14:30:45,123 - MainThread - crustacean.models.binary_classifier - INFO - Loading model from processing/binary_classifier/save/DS1_A_200_128.tflite
2024-11-12 14:30:46,234 - AnalysisThread - crustacean.core.realtime_pipeline - DEBUG - Processing 30 frames starting at frame 120
2024-11-12 14:30:47,345 - ObjectDetectorThread - crustacean.models.object_detector - WARNING - Low confidence detection: 0.68
```

**Design Decisions**:
- Use Python's built-in `logging` module (no external dependencies)
- RotatingFileHandler for automatic log rotation
- Thread names in format for debugging multi-threaded code
- Separate loggers per module for granular control
- Console and file handlers can be independently enabled

### 3. Base Model Interface

**Module**: `crustacean/models/base_model.py`

**Purpose**: Abstract interface for all ML models, eliminating duplication

**Key Classes**:
```python
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, config: Config, preload: bool = False):
        """
        Initialize model with configuration
        
        Args:
            config: Configuration object
            preload: If True, load model immediately
        """
        self.config = config
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger = get_logger(self.__class__.__name__)
        
        if preload:
            self.load()
    
    @abstractmethod
    def load(self) -> None:
        """Load the TFLite model and allocate tensors"""
        
    @abstractmethod
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input data for model"""
        
    @abstractmethod
    def postprocess(self, output_data: np.ndarray) -> Any:
        """Postprocess model output"""
        
    def predict(self, input_data: Any) -> Any:
        """
        Run inference on input data
        
        Args:
            input_data: Raw input (frame, video, etc.)
            
        Returns:
            Processed output
        """
        if self.interpreter is None:
            raise ModelNotLoadedError(f"{self.__class__.__name__} not loaded")
            
        preprocessed = self.preprocess(input_data)
        
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            preprocessed.astype(np.float32)
        )
        self.interpreter.invoke()
        
        raw_output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return self.postprocess(raw_output)
    
    def unload(self) -> None:
        """Release model resources"""
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger.info(f"{self.__class__.__name__} unloaded")
    
    def __enter__(self):
        """Context manager support"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.unload()
```

**Design Decisions**:
- Template method pattern: `predict()` orchestrates load/preprocess/infer/postprocess
- Context manager support for automatic cleanup
- Lazy loading by default, optional preloading
- Consistent error handling across all models
- Logging integrated at base level


### 4. Model Implementations

Each model extends `BaseModel` and implements the abstract methods:

**Binary Classifier** (`crustacean/models/binary_classifier.py`):
```python
class BinaryClassifier(BaseModel):
    """Detects crustacean presence in video frames"""
    
    def load(self) -> None:
        model_path = self.config.get('models.binary_classifier.path')
        self.interpreter = tflite.Interpreter(model_path=model_path)
        # ... configure tensors
        
    def preprocess(self, video: cv2.VideoCapture) -> np.ndarray:
        # Extract frames, convert to grayscale, resize
        
    def postprocess(self, predictions: np.ndarray) -> np.ndarray:
        # Apply smoothing and rectification
        return self._apply_smoothing(predictions)
    
    def _apply_smoothing(self, preds: np.ndarray) -> np.ndarray:
        """Rectangle smoothing and rectification"""
        # Existing logic from binary_classifier_util.py
```

**Frame Selector** (`crustacean/models/frame_selector.py`):
```python
class FrameSelector(BaseModel):
    """Selects highest quality frames from video segments"""
    
    def __init__(self, config: Config, preload: bool = False):
        super().__init__(config, preload)
        self.top_interpreter = None
        self.bottom_interpreter = None
        
    def load(self) -> None:
        # Load both top and bottom models
        
    def predict(self, signal: np.ndarray, video: cv2.VideoCapture) -> List[List[int]]:
        """
        Select best frames based on binary classifier signal
        
        Returns:
            [top_frame_indices, bottom_frame_indices]
        """
```

**Object Detector** (`crustacean/models/object_detector.py`):
```python
class ObjectDetector(BaseModel):
    """Detects and localizes crustaceans in frames"""
    
    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Detect crustacean and return cropped ROI
        
        Returns:
            (cropped_roi, confidence, class_index)
        """
```

**Keypoint Detector** (`crustacean/models/keypoint_detector.py`):
```python
class KeypointDetector(BaseModel):
    """Detects anatomical keypoints in cropped ROIs"""
    
    def predict(self, roi_frames: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in ROI frames
        
        Returns:
            Array of shape (n_frames, 14) with x,y coordinates
        """
```

### 5. Pipeline Architecture

**Base Pipeline** (`crustacean/core/pipeline.py`):
```python
class Pipeline(ABC):
    """Abstract base class for all pipeline modes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.models = {}
        
    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline"""
        
    def load_models(self, preload: bool = False) -> None:
        """Load all required models"""
        self.models['bc'] = BinaryClassifier(self.config, preload)
        self.models['fs'] = FrameSelector(self.config, preload)
        self.models['od'] = ObjectDetector(self.config, preload)
        self.models['kd'] = KeypointDetector(self.config, preload)
        
    def cleanup(self) -> None:
        """Release all resources"""
        for model in self.models.values():
            model.unload()
```

**Offline Pipeline** (`crustacean/core/offline_pipeline.py`):
```python
class OfflinePipeline(Pipeline):
    """Batch processing of video files"""
    
    def __init__(self, config: Config, video_dir: str):
        super().__init__(config)
        self.video_dir = video_dir
        
    def run(self) -> None:
        """Process all videos in directory"""
        self.load_models(preload=False)  # Load/unload per video
        
        for video_path in self._get_video_files():
            self._process_video(video_path)
            
    def _process_video(self, video_path: str) -> None:
        """Process single video through 4-stage pipeline"""
        # BC -> FS -> OD -> KD
```


**Real-time Pipeline** (`crustacean/core/realtime_pipeline.py`):
```python
class RealtimePipeline(Pipeline):
    """Live camera processing with multi-threading"""
    
    def __init__(self, config: Config, display_mode: bool = False):
        super().__init__(config)
        self.display_mode = display_mode
        self.camera = None
        self.threads = {}
        self.queues = {}
        self.executor = None
        
    def run(self) -> None:
        """Main real-time processing loop"""
        try:
            self._initialize()
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        finally:
            self._shutdown()
            
    def _initialize(self) -> None:
        """Initialize camera, models, and threads"""
        self.camera = self._create_camera()
        self.load_models(preload=True)  # Keep in memory
        self._start_threads()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('realtime.max_save_threads', 2)
        )
        
    def _main_loop(self) -> None:
        """Main processing loop with motion detection"""
        frame_counter = 0
        collecting = False
        collected_frames = []
        
        while True:
            frame = self.camera.read()
            
            if self._should_process_frame(frame_counter):
                if self._detect_motion(frame):
                    collecting = True
                    
            if collecting:
                collected_frames.append(frame)
                if len(collected_frames) >= self.config.get('realtime.frames_to_collect'):
                    self._submit_for_analysis(collected_frames)
                    collecting = False
                    collected_frames = []
                    
            if self.display_mode:
                self._render_frame(frame)
                
            frame_counter += 1
            
    def _render_frame(self, frame: np.ndarray) -> None:
        """Render frame with overlays (only in display mode)"""
        # Display-specific code isolated here
        
    def _shutdown(self) -> None:
        """Graceful shutdown"""
        self.logger.info("Shutting down...")
        
        # Stop threads
        for thread in self.threads.values():
            thread.stop()
            thread.join(timeout=5)
            
        # Wait for saves to complete
        if self.executor:
            self.executor.shutdown(wait=True, timeout=10)
            
        # Release camera
        if self.camera:
            self.camera.release()
            
        # Unload models
        self.cleanup()
        
        self.logger.info("Shutdown complete")
```

**Design Decisions**:
- Single implementation for headless and display modes
- Display code isolated in `_render_frame()` method
- Graceful shutdown with timeouts
- ThreadPoolExecutor for controlled save operations
- All configuration from Config object

### 6. Thread Management

**Analysis Thread** (`crustacean/threads/analysis_thread.py`):
```python
class AnalysisThread(Thread):
    """Processes frames through BC and FS"""
    
    def __init__(self, 
                 analysis_queue: Queue,
                 detection_queue: Queue,
                 bc_model: BinaryClassifier,
                 fs_model: FrameSelector):
        super().__init__(name="AnalysisThread")
        self.analysis_queue = analysis_queue
        self.detection_queue = detection_queue
        self.bc = bc_model
        self.fs = fs_model
        self.running = True
        self.logger = get_logger(__name__)
        
    def run(self) -> None:
        """Main thread loop"""
        try:
            while self.running:
                try:
                    frames, start_idx = self.analysis_queue.get(timeout=2)
                    self._process_frames(frames, start_idx)
                except queue.Empty:
                    continue
        except Exception as e:
            self.logger.exception(f"Analysis thread failed: {e}")
        finally:
            self.logger.info("Analysis thread stopped")
            
    def _process_frames(self, frames: List[np.ndarray], start_idx: int) -> None:
        """Process frame batch through BC and FS"""
        # Create temp video, run BC, run FS, select best frame
        
    def stop(self) -> None:
        """Signal thread to stop"""
        self.running = False
```

Similar structure for `DetectionThread` and `SaveThread`.

**Design Decisions**:
- Each thread is self-contained with its own error handling
- Threads receive model instances (already loaded)
- Clean stop mechanism with timeout
- Comprehensive logging for debugging


### 7. Camera Abstraction

**Module**: `crustacean/camera/`

**Purpose**: Abstract camera interface supporting multiple backends

**Base Camera** (`crustacean/camera/base_camera.py`):
```python
class BaseCamera(ABC):
    """Abstract camera interface"""
    
    @abstractmethod
    def open(self) -> bool:
        """Open camera connection"""
        
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read next frame"""
        
    @abstractmethod
    def release(self) -> None:
        """Release camera resources"""
        
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is open"""
```

**GStreamer Camera** (`crustacean/camera/gstreamer_camera.py`):
```python
class GStreamerCamera(BaseCamera):
    """CSI camera via GStreamer pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = self._build_pipeline()
        self.capture = None
        
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline from config"""
        width = self.config.get('camera.width')
        height = self.config.get('camera.height')
        fps = self.config.get('camera.framerate')
        rotation = self.config.get('camera.rotation')
        
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1 ! "
            f"nvvidconv ! videoflip method=rotate-{rotation} ! "
            f"video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=2 sync=false"
        )
```

**Design Decisions**:
- Factory pattern for camera creation based on config
- Easy to add USB, RTSP, or file-based cameras
- Configuration-driven pipeline construction

### 8. Exception Hierarchy

**Module**: `crustacean/utils/exceptions.py`

```python
class CrustaceanError(Exception):
    """Base exception for all crustacean monitoring errors"""
    pass

class ConfigurationError(CrustaceanError):
    """Configuration file or validation errors"""
    pass

class ModelLoadError(CrustaceanError):
    """Model loading failures"""
    pass

class ModelNotLoadedError(CrustaceanError):
    """Attempting inference without loaded model"""
    pass

class CameraInitError(CrustaceanError):
    """Camera initialization failures"""
    pass

class InferenceError(CrustaceanError):
    """Model inference failures"""
    pass

class ThreadError(CrustaceanError):
    """Thread management errors"""
    pass
```

**Design Decisions**:
- Hierarchy allows catching specific or general errors
- Clear error names indicate failure point
- All inherit from base `CrustaceanError`

### 9. Monitoring System

**Module**: `crustacean/monitoring/`

**Hardware Detector** (`crustacean/monitoring/hardware_detector.py`):
```python
def detect_hardware() -> str:
    """
    Auto-detect hardware platform
    
    Returns:
        'jetson', 'raspberry_pi', or 'generic'
    """
    machine = platform.machine()
    if machine == "aarch64":
        # Check for Jetson-specific files
        if os.path.exists('/etc/nv_tegra_release'):
            return 'jetson'
    elif machine == "armv7l":
        return 'raspberry_pi'
    return 'generic'

def create_monitor(config: Config) -> BaseMonitor:
    """Factory function to create appropriate monitor"""
    hardware = detect_hardware()
    
    if hardware == 'jetson':
        return JetsonMonitor(config)
    elif hardware == 'raspberry_pi':
        return RaspberryPiMonitor(config)
    else:
        return GenericMonitor(config)
```

**Base Monitor** (`crustacean/monitoring/base_monitor.py`):
```python
class BaseMonitor(Thread):
    """Base hardware monitoring class"""
    
    def __init__(self, config: Config, output_file: str):
        super().__init__(name="MonitorThread")
        self.config = config
        self.output_file = output_file
        self.interval = 2
        self.stop_event = Event()
        self.logger = get_logger(__name__)
        
    def run(self) -> None:
        """Main monitoring loop"""
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.get_fieldnames())
            writer.writeheader()
            
            while not self.stop_event.wait(self.interval):
                metrics = self.collect_metrics()
                writer.writerow(metrics)
                
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect hardware metrics"""
        
    def get_common_metrics(self) -> Dict[str, Any]:
        """Metrics available on all platforms"""
        return {
            'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S"),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'ram_percent': psutil.virtual_memory().percent
        }
```

**Design Decisions**:
- Auto-detection eliminates manual configuration
- Factory pattern for monitor creation
- Common metrics in base class
- Platform-specific metrics in subclasses
- Graceful degradation if platform APIs unavailable


## Data Models

### Configuration Data Model

```python
@dataclass
class ModelConfig:
    path: str
    input_width: int
    input_height: int
    
@dataclass
class BinaryClassifierConfig(ModelConfig):
    batch_size: int = 1
    smoothing_gamma: int = 20
    rectify_theta: float = 0.5
    
@dataclass
class ObjectDetectorConfig(ModelConfig):
    confidence_threshold: float = 0.75
    fixed_crop_width: int = 539
    fixed_crop_height: int = 561
    
@dataclass
class RealtimeConfig:
    motion_detection_threshold: int = 15
    detection_cooldown: int = 3
    frames_to_collect: int = 30
    process_interval: int = 30
    max_save_threads: int = 2
    
@dataclass
class CameraConfig:
    type: str = "csi"
    width: int = 1280
    height: int = 720
    framerate: int = 45
    rotation: int = 180
    device: str = "/dev/video0"
```

### Detection Data Model

```python
@dataclass
class Detection:
    """Represents a single crustacean detection"""
    timestamp: datetime
    frame_number: int
    confidence: float
    class_index: int  # 0=crab, 1=lobster
    keypoints: np.ndarray  # Shape (7, 2)
    frame_path: str
    csv_path: str
    
    def save(self, output_dir: str) -> None:
        """Save detection to disk"""
        
    @classmethod
    def load(cls, detection_dir: str) -> 'Detection':
        """Load detection from disk"""
```

## Error Handling

### Error Handling Strategy

1. **Model Loading Errors**:
   - Catch at model initialization
   - Log full path and error details
   - Raise `ModelLoadError` with context
   - Fail fast - don't continue without models

2. **Inference Errors**:
   - Catch per-frame in real-time mode
   - Log warning with frame number
   - Skip frame and continue processing
   - Track error rate, shutdown if too high

3. **Camera Errors**:
   - Retry connection 3 times with backoff
   - Log each attempt
   - Raise `CameraInitError` if all fail
   - Provide troubleshooting hints in error message

4. **Thread Errors**:
   - Catch all exceptions in thread run() method
   - Log full stack trace
   - Perform cleanup (close files, release resources)
   - Set error flag for main thread to detect

5. **Configuration Errors**:
   - Validate on load
   - Provide specific error messages (e.g., "models.binary_classifier.path is required")
   - Suggest fixes when possible
   - Create default config if missing

### Error Recovery

```python
class RealtimePipeline(Pipeline):
    def _main_loop(self) -> None:
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while True:
            try:
                frame = self.camera.read()
                if frame is None:
                    raise CameraError("Failed to read frame")
                    
                # Process frame...
                consecutive_errors = 0  # Reset on success
                
            except InferenceError as e:
                self.logger.warning(f"Inference failed: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Too many consecutive errors, shutting down")
                    break
                    
            except CameraError as e:
                self.logger.error(f"Camera error: {e}")
                if not self._reconnect_camera():
                    break
```

## Testing Strategy

### Unit Tests

**Test Structure**:
```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_logging.py
│   ├── test_models/
│   │   ├── test_binary_classifier.py
│   │   ├── test_frame_selector.py
│   │   ├── test_object_detector.py
│   │   └── test_keypoint_detector.py
│   ├── test_camera.py
│   └── test_monitoring.py
```

**Mock Strategy**:
- Mock TFLite interpreter for model tests
- Mock cv2.VideoCapture for camera tests
- Use small test fixtures (10-frame videos)
- Mock hardware APIs (jtop, gpiozero)

**Example Test**:
```python
# tests/unit/test_models/test_binary_classifier.py
import pytest
from unittest.mock import Mock, patch
from crustacean.models.binary_classifier import BinaryClassifier

@pytest.fixture
def mock_config():
    config = Mock()
    config.get.side_effect = lambda key, default=None: {
        'models.binary_classifier.path': 'test_model.tflite',
        'models.binary_classifier.input_width': 320,
        'models.binary_classifier.input_height': 180,
    }.get(key, default)
    return config

@patch('tflite_runtime.interpreter.Interpreter')
def test_binary_classifier_load(mock_interpreter, mock_config):
    bc = BinaryClassifier(mock_config)
    bc.load()
    
    assert bc.interpreter is not None
    mock_interpreter.assert_called_once_with(model_path='test_model.tflite')

def test_binary_classifier_predict_without_load(mock_config):
    bc = BinaryClassifier(mock_config)
    
    with pytest.raises(ModelNotLoadedError):
        bc.predict(Mock())
```

### Integration Tests

**Test Structure**:
```
tests/
├── integration/
│   ├── test_offline_pipeline.py
│   ├── test_realtime_pipeline.py
│   └── test_end_to_end.py
├── fixtures/
│   ├── test_video.mp4
│   ├── test_frames/
│   └── mock_models/
```

**Integration Test Approach**:
- Use actual small TFLite models (or mock models with correct I/O shapes)
- Test full pipeline with test video
- Verify output files created correctly
- Test thread coordination
- Test graceful shutdown

**Example Integration Test**:
```python
# tests/integration/test_offline_pipeline.py
def test_offline_pipeline_processes_video(tmp_path, test_video, test_config):
    """Test complete offline pipeline"""
    output_dir = tmp_path / "output"
    
    pipeline = OfflinePipeline(test_config, video_dir=str(test_video.parent))
    pipeline.run()
    
    # Verify outputs
    assert (output_dir / "extracted_frames").exists()
    # Check for expected number of detections
```

### Test Coverage Goals

- **Core modules**: 80%+ coverage
- **Models**: 70%+ coverage (some TFLite internals hard to test)
- **Pipelines**: 75%+ coverage
- **Utils**: 90%+ coverage (config, logging, exceptions)

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=crustacean --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```


## Performance Considerations

### Profiling Infrastructure

**Module**: `crustacean/utils/profiling.py`

```python
import time
from contextlib import contextmanager
from typing import Dict, List
import cProfile
import pstats

class PerformanceProfiler:
    """Tracks performance metrics across pipeline"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.logger = get_logger(__name__)
        
    @contextmanager
    def profile_section(self, name: str):
        """Context manager for timing code sections"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings.setdefault(name, []).append(elapsed)
            self.logger.debug(f"{name}: {elapsed:.3f}s")
            
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times)
            }
        return summary
        
    def print_summary(self) -> None:
        """Print formatted timing summary"""
        summary = self.get_summary()
        print("\n=== Performance Summary ===")
        for name, stats in summary.items():
            print(f"{name}:")
            print(f"  Mean: {stats['mean']:.3f}s")
            print(f"  Std:  {stats['std']:.3f}s")
            print(f"  Min:  {stats['min']:.3f}s")
            print(f"  Max:  {stats['max']:.3f}s")
            print(f"  Total: {stats['total']:.3f}s")
```

**Usage in Pipeline**:
```python
class OfflinePipeline(Pipeline):
    def __init__(self, config: Config, video_dir: str, profile: bool = False):
        super().__init__(config)
        self.profiler = PerformanceProfiler() if profile else None
        
    def _process_video(self, video_path: str) -> None:
        if self.profiler:
            with self.profiler.profile_section("Binary Classifier"):
                signal = self.models['bc'].predict(video)
                
            with self.profiler.profile_section("Frame Selector"):
                indices = self.models['fs'].predict(signal, video)
        else:
            signal = self.models['bc'].predict(video)
            indices = self.models['fs'].predict(signal, video)
```

### Memory Management

**Strategies**:
1. **Explicit cleanup**: Call `del` on large arrays after use
2. **Garbage collection**: Force `gc.collect()` after processing each video
3. **Model unloading**: Unload models when not needed (offline mode)
4. **Frame batching**: Process frames in batches to limit memory
5. **Queue size limits**: Prevent unbounded queue growth

**Example**:
```python
def _process_video(self, video_path: str) -> None:
    signal = self.models['bc'].predict(video)
    indices = self.models['fs'].predict(signal, video)
    
    # Explicitly free memory
    del signal
    gc.collect()
    
    roi_frames = self.models['od'].predict(frames)
    keypoints = self.models['kd'].predict(roi_frames)
    
    del roi_frames
    gc.collect()
```

### Threading Optimization

**Queue Management**:
- `maxsize=1` for analysis and detection queues (backpressure)
- Prevents memory buildup if processing is slow
- Main thread blocks if queue full (natural rate limiting)

**Thread Pool Sizing**:
- SaveDetection: 2 workers (I/O bound)
- Analysis: 1 thread (CPU bound, model inference)
- Detection: 1 thread (CPU bound, model inference)

**Design Rationale**:
- More threads doesn't help CPU-bound tasks on Jetson
- Limited to 2 save threads to avoid I/O contention
- Single analysis/detection threads simplify debugging

## Migration Strategy

### Phase 1: Foundation (Week 1)

**Goal**: Establish core infrastructure without breaking existing code

**Tasks**:
1. Create new package structure (`crustacean/`)
2. Implement configuration system
3. Implement logging system
4. Create base model class
5. Add custom exceptions
6. Write unit tests for utils

**Validation**:
- Config loads from YAML
- Logging outputs to console and file
- Tests pass

### Phase 2: Model Refactor (Week 2)

**Goal**: Migrate model utilities to new structure

**Tasks**:
1. Implement BinaryClassifier extending BaseModel
2. Implement FrameSelector extending BaseModel
3. Implement ObjectDetector extending BaseModel
4. Implement KeypointDetector extending BaseModel
5. Write unit tests for each model
6. Keep old `*_util.py` files for backward compatibility

**Validation**:
- New models produce same output as old utils
- Unit tests pass
- Integration test with test video

### Phase 3: Pipeline Refactor (Week 3)

**Goal**: Migrate pipeline implementations

**Tasks**:
1. Implement base Pipeline class
2. Implement OfflinePipeline
3. Implement RealtimePipeline (unified)
4. Implement thread classes
5. Implement camera abstraction
6. Write integration tests

**Validation**:
- Offline pipeline processes test video
- Real-time pipeline runs in headless mode
- Real-time pipeline runs in display mode
- All tests pass

### Phase 4: Monitoring & Polish (Week 4)

**Goal**: Complete monitoring system and finalize

**Tasks**:
1. Implement monitoring system
2. Add profiling support
3. Create entry point scripts
4. Write comprehensive documentation
5. Create migration guide
6. Final testing on Jetson Nano

**Validation**:
- All modes work on Jetson Nano
- Monitoring collects metrics
- Documentation complete
- Performance comparable or better than original

### Backward Compatibility

During migration, maintain compatibility:

```python
# Old interface (deprecated)
import processing.binary_classifier_util as bc
bc.process(video)

# New interface
from crustacean.models import BinaryClassifier
bc = BinaryClassifier(config)
bc.predict(video)
```

Create shim modules:
```python
# processing/binary_classifier_util.py (compatibility shim)
import warnings
from crustacean.models import BinaryClassifier
from crustacean.utils.config import Config

warnings.warn(
    "processing.binary_classifier_util is deprecated, "
    "use crustacean.models.BinaryClassifier",
    DeprecationWarning
)

_config = Config.load()
_model = BinaryClassifier(_config)

def process(video):
    """Deprecated: Use BinaryClassifier.predict()"""
    return _model.predict(video)
```

### Rollback Plan

If issues arise:
1. Keep old code in `legacy/` directory
2. Git tags for each phase
3. Can revert to any phase
4. Shim modules allow gradual migration

## Documentation

### Code Documentation

**Docstring Format** (Google Style):
```python
def predict(self, input_data: Any) -> Any:
    """
    Run inference on input data.
    
    This method preprocesses the input, runs model inference,
    and postprocesses the output.
    
    Args:
        input_data: Raw input data (frame, video, etc.)
            For BinaryClassifier: cv2.VideoCapture object
            For ObjectDetector: numpy array of shape (H, W, 3)
            
    Returns:
        Processed model output. Type depends on model:
            BinaryClassifier: numpy array of binary values (0 or 1)
            ObjectDetector: tuple of (roi, confidence, class_index)
            
    Raises:
        ModelNotLoadedError: If model not loaded before calling predict
        InferenceError: If inference fails
        
    Example:
        >>> bc = BinaryClassifier(config)
        >>> bc.load()
        >>> signal = bc.predict(video_capture)
        >>> print(signal.shape)
        (300,)
    """
```

### User Documentation

**README Updates**:
- Installation with new requirements
- Configuration file documentation
- Usage examples for new CLI
- Migration guide from old code
- Troubleshooting section

**Configuration Reference**:
- Document every config option
- Provide examples for common scenarios
- Explain defaults and valid ranges

**API Documentation**:
- Auto-generate with Sphinx
- Host on GitHub Pages
- Include tutorials and examples

## Summary

This design provides:

1. **Elimination of duplication**: Single implementation per model, unified real-time pipeline
2. **Production readiness**: Logging, error handling, graceful shutdown
3. **Maintainability**: Clear structure, type hints, comprehensive tests
4. **Flexibility**: Configuration-driven, extensible architecture
5. **Performance**: Profiling tools, memory management, optimized threading

The phased migration approach allows incremental implementation while maintaining a working system throughout the refactor.
