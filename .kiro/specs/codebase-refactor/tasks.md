# Implementation Plan

This implementation plan breaks down the comprehensive refactor into discrete, manageable coding tasks. Each task builds incrementally on previous tasks, following the phased approach: Foundation → Model Refactor → Pipeline Refactor → Monitoring & Polish.

## Phase 1: Foundation (Week 1)

- [x] 1. Set up new package structure
  - Create `crustacean/` package directory with `__init__.py`
  - Create subdirectories: `core/`, `models/`, `monitoring/`, `camera/`, `utils/`, `threads/`
  - Create `config/` directory for configuration files
  - Create `tests/` directory with `unit/` and `integration/` subdirectories
  - Create `scripts/` directory for entry point scripts
  - Add `__init__.py` files to all package directories
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 2. Implement configuration system
  - [ ] 2.1 Create default configuration YAML file
    - Define all model configurations (paths, dimensions, thresholds)
    - Define real-time processing parameters
    - Define camera configuration options
    - Define output directory paths
    - Define logging configuration
    - Save as `config/default_config.yaml`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ] 2.2 Implement Config class
    - Write `crustacean/utils/config.py`
    - Implement `Config.load()` class method to load YAML
    - Implement `get()` method with dot notation support
    - Implement default value handling
    - Implement environment variable override support
    - Add error handling for missing/invalid config files
    - _Requirements: 2.1, 2.6, 2.7_
  
  - [ ] 2.3 Write configuration tests
    - Test loading valid config file
    - Test loading with missing file (creates default)
    - Test dot notation access
    - Test environment variable overrides
    - Test invalid YAML handling
    - _Requirements: 9.3_

- [ ] 3. Implement logging system
  - [ ] 3.1 Create logging setup module
    - Write `crustacean/utils/logging_setup.py`
    - Implement `setup_logging()` function
    - Configure console and file handlers
    - Configure RotatingFileHandler with 10MB limit, 5 backups
    - Set log format with timestamp, thread name, module, level
    - Implement `get_logger()` helper function
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [ ] 3.2 Write logging tests
    - Test logger creation
    - Test log output format
    - Test file rotation
    - Test log level filtering
    - _Requirements: 9.4_

- [ ] 4. Create custom exception hierarchy
  - Write `crustacean/utils/exceptions.py`
  - Define `CrustaceanError` base exception
  - Define `ConfigurationError` for config issues
  - Define `ModelLoadError` for model loading failures
  - Define `ModelNotLoadedError` for inference without loaded model
  - Define `CameraInitError` for camera failures
  - Define `InferenceError` for inference failures
  - Define `ThreadError` for thread management issues
  - Add docstrings explaining when each exception is raised
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 5. Implement base model interface
  - [ ] 5.1 Create BaseModel abstract class
    - Write `crustacean/models/base_model.py`
    - Define `__init__()` accepting config and preload flag
    - Define abstract `load()` method
    - Define abstract `preprocess()` method
    - Define abstract `postprocess()` method
    - Implement concrete `predict()` method orchestrating the pipeline
    - Implement `unload()` method for resource cleanup
    - Implement context manager methods (`__enter__`, `__exit__`)
    - Add comprehensive docstrings with type hints
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 12.1, 12.2_
  
  - [ ] 5.2 Write BaseModel tests
    - Create mock model extending BaseModel
    - Test initialization with preload=True and False
    - Test predict() workflow
    - Test context manager usage
    - Test error handling for predict() without load()
    - _Requirements: 9.1_

- [ ] 6. Create dependency files
  - Create `requirements.txt` with pinned versions (tflite-runtime, numpy, opencv-python, Pillow, psutil, PyYAML)
  - Create `requirements-dev.txt` (pytest, pytest-cov, black, flake8, mypy)
  - Create `requirements-jetson.txt` (jetson-stats)
  - Document Python version requirement (3.9+)
  - _Requirements: 13.1, 13.2, 13.3, 13.5_

## Phase 2: Model Refactor (Week 2)

- [ ] 7. Implement BinaryClassifier model
  - [ ] 7.1 Create BinaryClassifier class
    - Write `crustacean/models/binary_classifier.py`
    - Extend BaseModel
    - Implement `load()` to load TFLite model
    - Implement `preprocess()` for video frame extraction and grayscale conversion
    - Implement `postprocess()` with smoothing and rectification logic
    - Port `rectangle_smooth()` and `rectify()` helper functions
    - Add logging at key points
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 7.2 Write BinaryClassifier tests
    - Test model loading
    - Test preprocessing with mock video
    - Test smoothing and rectification
    - Test full predict() pipeline
    - _Requirements: 9.1_

- [ ] 8. Implement FrameSelector model
  - [ ] 8.1 Create FrameSelector class
    - Write `crustacean/models/frame_selector.py`
    - Extend BaseModel
    - Implement `load()` to load both top and bottom TFLite models
    - Implement `preprocess()` for frame rescaling and reshaping
    - Implement `postprocess()` to return frame indices
    - Implement `predict()` override to handle signal and video inputs
    - Port contig detection logic
    - Add logging for segment processing
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 8.2 Write FrameSelector tests
    - Test loading both models
    - Test frame quality prediction
    - Test contig detection logic
    - Test best frame selection
    - _Requirements: 9.1_

- [ ] 9. Implement ObjectDetector model
  - [ ] 9.1 Create ObjectDetector class
    - Write `crustacean/models/object_detector.py`
    - Extend BaseModel
    - Implement `load()` to load TFLite model
    - Implement `preprocess()` for padding and resizing
    - Implement `postprocess()` with NMS and cropping logic
    - Port fixed box size cropping logic
    - Return tuple of (roi, confidence, class_index)
    - Add confidence threshold checking
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 9.2 Write ObjectDetector tests
    - Test model loading
    - Test preprocessing (padding, resizing)
    - Test NMS application
    - Test ROI cropping
    - Test confidence thresholding
    - _Requirements: 9.1_

- [ ] 10. Implement KeypointDetector model
  - [ ] 10.1 Create KeypointDetector class
    - Write `crustacean/models/keypoint_detector.py`
    - Extend BaseModel
    - Implement `load()` to load TFLite model
    - Implement `preprocess()` for ROI reshaping
    - Implement `postprocess()` to return keypoint coordinates
    - Handle batch processing of multiple ROI frames
    - Return array of shape (n_frames, 14)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 10.2 Write KeypointDetector tests
    - Test model loading
    - Test single frame processing
    - Test batch processing
    - Test output shape validation
    - _Requirements: 9.1_

- [ ] 11. Create model package exports
  - Update `crustacean/models/__init__.py`
  - Export all model classes
  - Add package-level docstring
  - _Requirements: 12.1_

## Phase 3: Pipeline Refactor (Week 3)

- [ ] 12. Implement base Pipeline class
  - Write `crustacean/core/pipeline.py`
  - Create abstract Pipeline class
  - Implement `__init__()` accepting config
  - Implement `load_models()` method to instantiate all models
  - Implement `cleanup()` method to unload all models
  - Define abstract `run()` method
  - Add logging initialization
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 13. Implement OfflinePipeline
  - [ ] 13.1 Create OfflinePipeline class
    - Write `crustacean/core/offline_pipeline.py`
    - Extend Pipeline base class
    - Implement `__init__()` accepting config and video directory
    - Implement `run()` to process all videos in directory
    - Implement `_process_video()` for single video processing
    - Implement 4-stage pipeline: BC → FS → OD → KD
    - Add frame extraction and saving logic
    - Add completed files tracking
    - Integrate profiling support (optional)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 10.2, 10.3, 10.4_
  
  - [ ] 13.2 Write OfflinePipeline integration test
    - Test processing test video end-to-end
    - Verify output files created
    - Verify keypoint CSV format
    - Test completed files tracking
    - _Requirements: 9.2_

- [ ] 14. Implement camera abstraction
  - [ ] 14.1 Create BaseCamera interface
    - Write `crustacean/camera/base_camera.py`
    - Define abstract `open()` method
    - Define abstract `read()` method
    - Define abstract `release()` method
    - Define abstract `is_opened()` method
    - _Requirements: 8.3_
  
  - [ ] 14.2 Create GStreamerCamera implementation
    - Write `crustacean/camera/gstreamer_camera.py`
    - Extend BaseCamera
    - Implement `_build_pipeline()` from config parameters
    - Implement `open()` to initialize cv2.VideoCapture with pipeline
    - Implement `read()` to return frames
    - Implement `release()` to close capture
    - Add error handling for camera initialization
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 14.3 Create camera factory function
    - Add `create_camera()` function in `crustacean/camera/__init__.py`
    - Auto-select camera type based on config
    - Return appropriate camera instance
    - _Requirements: 8.3_

- [ ] 15. Implement thread classes
  - [ ] 15.1 Create AnalysisThread
    - Write `crustacean/threads/analysis_thread.py`
    - Extend Thread
    - Implement `__init__()` accepting queues and models
    - Implement `run()` with main processing loop
    - Implement `_process_frames()` for BC and FS processing
    - Create temporary video from frame array
    - Add error handling and logging
    - Implement `stop()` method
    - _Requirements: 5.4, 5.5, 6.1, 6.2_
  
  - [ ] 15.2 Create DetectionThread
    - Write `crustacean/threads/detection_thread.py`
    - Extend Thread
    - Implement `__init__()` accepting queues and OD model
    - Implement `run()` with main processing loop
    - Process frames through object detector
    - Add error handling and logging
    - Implement `stop()` method
    - _Requirements: 5.4, 5.5, 6.1, 6.2_
  
  - [ ] 15.3 Create SaveThread
    - Write `crustacean/threads/save_thread.py`
    - Create `save_detection()` function (not a thread class)
    - Accept frame, roi, confidence, frame_number, config
    - Load keypoint detector
    - Generate unique directory and filenames
    - Save frame as JPG
    - Process ROI through keypoint detector
    - Save keypoints as CSV
    - Unload keypoint detector
    - Add comprehensive error handling
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 16. Implement RealtimePipeline
  - [ ] 16.1 Create unified RealtimePipeline class
    - Write `crustacean/core/realtime_pipeline.py`
    - Extend Pipeline base class
    - Implement `__init__()` accepting config and display_mode flag
    - Implement `_initialize()` to set up camera, models, threads, executor
    - Implement `_main_loop()` with motion detection and frame collection
    - Implement `_detect_motion()` using frame differencing
    - Implement `_should_process_frame()` for interval checking
    - Implement `_submit_for_analysis()` to queue frames
    - Implement `_handle_detection_results()` to process OD output
    - Implement `_render_frame()` for display mode (isolated method)
    - Implement `_shutdown()` for graceful cleanup
    - Use ThreadPoolExecutor for save operations
    - Add cooldown mechanism to prevent duplicate detections
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4, 15.1, 15.2, 15.3, 15.4, 15.5_
  
  - [ ] 16.2 Implement graceful shutdown
    - Handle KeyboardInterrupt in main loop
    - Stop all threads with timeout
    - Wait for executor to complete saves
    - Release camera
    - Unload models
    - Log shutdown summary
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_
  
  - [ ] 16.3 Write RealtimePipeline integration test
    - Test initialization
    - Test motion detection
    - Test frame collection
    - Test thread coordination
    - Test graceful shutdown
    - _Requirements: 9.2_

## Phase 4: Monitoring & Polish (Week 4)

- [ ] 17. Implement monitoring system
  - [ ] 17.1 Create hardware detector
    - Write `crustacean/monitoring/hardware_detector.py`
    - Implement `detect_hardware()` function
    - Check platform.machine() and system files
    - Return 'jetson', 'raspberry_pi', or 'generic'
    - Implement `create_monitor()` factory function
    - _Requirements: 11.3_
  
  - [ ] 17.2 Create BaseMonitor class
    - Write `crustacean/monitoring/base_monitor.py`
    - Extend Thread
    - Implement `__init__()` accepting config and output file
    - Implement `run()` with CSV writing loop
    - Implement `get_common_metrics()` for cross-platform metrics
    - Define abstract `collect_metrics()` method
    - Implement `stop()` method
    - _Requirements: 11.1, 11.2, 11.4, 11.5_
  
  - [ ] 17.3 Create JetsonMonitor class
    - Write `crustacean/monitoring/jetson_monitor.py`
    - Extend BaseMonitor
    - Implement `collect_metrics()` using jtop
    - Collect CPU temp, GPU temp from jtop
    - Handle jtop initialization and cleanup
    - _Requirements: 11.1, 11.2, 11.4_
  
  - [ ] 17.4 Create RaspberryPiMonitor class
    - Write `crustacean/monitoring/pi_monitor.py`
    - Extend BaseMonitor
    - Implement `collect_metrics()` using gpiozero
    - Collect CPU temp from CPUTemperature
    - _Requirements: 11.1, 11.2, 11.4_
  
  - [ ] 17.5 Create GenericMonitor class
    - Write `crustacean/monitoring/generic_monitor.py`
    - Extend BaseMonitor
    - Implement `collect_metrics()` using only psutil
    - Provide fallback for platforms without specific support
    - _Requirements: 11.1, 11.2, 11.5_

- [ ] 18. Implement profiling tools
  - [ ] 18.1 Create PerformanceProfiler class
    - Write `crustacean/utils/profiling.py`
    - Implement `__init__()` to initialize timing storage
    - Implement `profile_section()` context manager
    - Implement `get_summary()` to calculate statistics
    - Implement `print_summary()` for formatted output
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 18.2 Integrate profiling into pipelines
    - Add optional profiler parameter to Pipeline classes
    - Wrap each stage with profiler.profile_section()
    - Print summary at end of run
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 19. Create entry point scripts
  - [ ] 19.1 Create offline pipeline script
    - Write `scripts/run_offline.py`
    - Parse command-line arguments (--config, --video-dir, --log-level, --profile)
    - Load configuration
    - Setup logging
    - Instantiate and run OfflinePipeline
    - Handle errors and exit codes
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  
  - [ ] 19.2 Create real-time pipeline script
    - Write `scripts/run_realtime.py`
    - Parse command-line arguments (--config, --display, --log-level, --profile)
    - Load configuration
    - Setup logging
    - Instantiate and run RealtimePipeline
    - Handle errors and exit codes
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  
  - [ ] 19.3 Create monitoring script
    - Write `scripts/run_monitoring.py`
    - Parse command-line arguments (--config, --video-dir, --output, --log-level)
    - Load configuration
    - Setup logging
    - Create monitor instance
    - Start monitor thread
    - Run OfflinePipeline
    - Stop monitor and save results
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 20. Write comprehensive documentation
  - [ ] 20.1 Update README
    - Document new installation process
    - Provide configuration file examples
    - Update usage examples for new scripts
    - Add troubleshooting section
    - Document breaking changes
    - _Requirements: 12.3, 12.4, 12.5_
  
  - [ ] 20.2 Create configuration reference
    - Document every configuration option
    - Provide default values
    - Explain valid ranges and types
    - Give examples for common scenarios
    - _Requirements: 12.3_
  
  - [ ] 20.3 Create migration guide
    - Document all breaking changes
    - Provide before/after code examples
    - Explain new project structure
    - Guide for updating existing deployments
    - _Requirements: 12.4_
  
  - [ ] 20.4 Generate API documentation
    - Set up Sphinx
    - Configure autodoc
    - Generate HTML documentation
    - _Requirements: 12.1, 12.2_

- [ ] 21. Final testing and validation
  - [ ] 21.1 Run full test suite
    - Execute all unit tests
    - Execute all integration tests
    - Verify coverage meets goals (70%+)
    - Fix any failing tests
    - _Requirements: 9.1, 9.2, 9.5_
  
  - [ ] 21.2 Test on Jetson Nano hardware
    - Deploy to Jetson Nano
    - Test offline pipeline with test videos
    - Test real-time pipeline in headless mode
    - Test real-time pipeline in display mode
    - Test monitoring system
    - Verify performance comparable to original
    - _Requirements: 9.2_
  
  - [ ] 21.3 Performance benchmarking
    - Run profiling on test videos
    - Compare timing to original implementation
    - Verify memory usage is acceptable
    - Document any performance differences
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 22. Create setup.py for package installation
  - Write `setup.py` with package metadata
  - Define entry points for scripts
  - Specify dependencies
  - Test installation with `pip install -e .`
  - _Requirements: 13.4_
