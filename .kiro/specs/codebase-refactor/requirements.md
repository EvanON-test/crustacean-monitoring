# Requirements Document

## Introduction

This document outlines the requirements for a comprehensive refactor of the Crustacean Monitoring System. The primary goals are to improve production readiness, eliminate code duplication, establish proper logging and configuration systems, and create a maintainable codebase structure. This refactor will follow a phased approach: Foundation → Consolidation → Testing → Deployment.

## Glossary

- **System**: The Crustacean Monitoring System
- **Pipeline**: The 4-stage processing workflow (Binary Classifier → Frame Selector → Object Detector → Keypoint Detector)
- **Real-time Mode**: Live camera processing with motion detection and multi-threaded architecture
- **Offline Mode**: Batch processing of pre-recorded video files
- **Model Util**: Utility module for loading and running inference with a specific ML model
- **Processing Thread**: Background thread that handles a specific stage of the pipeline
- **Detection**: A high-confidence identification of a crustacean with saved frame and keypoint data
- **Jetson Nano**: NVIDIA Jetson Nano 2GB edge device (primary deployment target)
- **TFLite Model**: TensorFlow Lite model file used for inference
- **Configuration File**: YAML file containing system parameters and settings
- **Logger**: Python logging system for structured output
- **Thread Pool**: Managed collection of worker threads with size limits

## Requirements

### Requirement 1: Logging System

**User Story:** As a developer, I want structured logging throughout the system, so that I can debug issues and monitor system behavior in production.

#### Acceptance Criteria

1. WHEN the System initializes, THE System SHALL configure a centralized logging system with configurable log levels
2. WHEN any module logs a message, THE System SHALL include timestamp, thread name, module name, and log level in the output
3. WHEN the System runs, THE System SHALL write logs to both console and rotating log files
4. WHEN log files exceed 10MB, THE System SHALL rotate logs and maintain up to 5 backup files
5. WHERE a module previously used print statements, THE System SHALL use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

### Requirement 2: Configuration Management

**User Story:** As a developer, I want all system parameters in a centralized configuration file, so that I can adjust settings without modifying code.

#### Acceptance Criteria

1. THE System SHALL load configuration from a YAML file at startup
2. THE Configuration File SHALL define all model paths, input dimensions, and confidence thresholds
3. THE Configuration File SHALL define all real-time processing parameters including motion detection threshold, cooldown period, and frame collection size
4. THE Configuration File SHALL define camera parameters including resolution, framerate, and rotation
5. THE Configuration File SHALL define all output directory paths
6. WHEN a configuration value is missing, THE System SHALL use documented default values
7. WHEN a configuration file does not exist, THE System SHALL create one with default values

### Requirement 3: Unified Model Utilities

**User Story:** As a developer, I want a single implementation for each model's processing logic, so that bug fixes and improvements are applied consistently.

#### Acceptance Criteria

1. THE System SHALL eliminate duplicate process() and process_realtime() functions in each Model Util
2. WHEN a Model Util processes data, THE System SHALL accept a parameter indicating whether to use preloaded models
3. WHEN using preloaded models, THE Model Util SHALL use globally cached model instances
4. WHEN not using preloaded models, THE Model Util SHALL load the model temporarily and release it after processing
5. THE System SHALL maintain a single code path for inference logic regardless of model loading strategy

### Requirement 4: Consolidated Real-time Pipeline

**User Story:** As a developer, I want a single real-time pipeline implementation with optional display mode, so that I can maintain one codebase instead of two nearly identical files.

#### Acceptance Criteria

1. THE System SHALL provide a base RealtimePipeline class containing all core processing logic
2. THE System SHALL provide a display mode parameter to enable or disable visualization
3. WHEN display mode is enabled, THE System SHALL render frames with detection overlays and metrics
4. WHEN display mode is disabled, THE System SHALL run in headless mode without GUI dependencies
5. THE System SHALL eliminate code duplication between realtime_pipeline.py and realtime_pipeline_demo.py
6. WHERE display-specific code exists, THE System SHALL isolate it in separate methods that can be overridden or skipped

### Requirement 5: Error Handling and Recovery

**User Story:** As a developer, I want robust error handling throughout the system, so that failures are logged clearly and the system can recover gracefully.

#### Acceptance Criteria

1. THE System SHALL define custom exception classes for model loading, camera initialization, and inference failures
2. WHEN a model fails to load, THE System SHALL log the error with full context and raise a ModelLoadError
3. WHEN camera initialization fails, THE System SHALL log the error with camera details and raise a CameraInitError
4. WHEN inference fails on a single frame, THE System SHALL log a warning and continue processing subsequent frames
5. WHEN a Processing Thread encounters an unhandled exception, THE System SHALL log the full stack trace and perform cleanup
6. WHEN the System shuts down, THE System SHALL ensure all threads are stopped and resources are released

### Requirement 6: Thread Pool Management

**User Story:** As a developer, I want controlled thread creation for detection saving, so that the system doesn't spawn unlimited threads during rapid detections.

#### Acceptance Criteria

1. THE System SHALL use a ThreadPoolExecutor for SaveDetection operations
2. THE System SHALL limit concurrent SaveDetection threads to a configurable maximum (default 2)
3. WHEN the Thread Pool is full, THE System SHALL queue additional save operations
4. WHEN the System shuts down, THE System SHALL wait for all queued save operations to complete
5. THE System SHALL log warnings if save operations are queuing due to high detection rates

### Requirement 7: Restructured Project Organization

**User Story:** As a developer, I want a clear and logical project structure, so that I can easily locate and modify code.

#### Acceptance Criteria

1. THE System SHALL organize code into logical packages: core, models, monitoring, utils, and config
2. THE System SHALL place all model-specific code in a models package with submodules for each model type
3. THE System SHALL place pipeline orchestration code in a core package
4. THE System SHALL place monitoring and metrics code in a monitoring package
5. THE System SHALL place shared utilities (logging setup, config loading) in a utils package
6. THE System SHALL maintain backward compatibility for model file paths and output directories

### Requirement 8: Parameterized Camera Configuration

**User Story:** As a developer, I want camera settings in the configuration file, so that I can support different cameras without code changes.

#### Acceptance Criteria

1. THE Configuration File SHALL define camera type (CSI, USB, RTSP)
2. THE Configuration File SHALL define camera resolution, framerate, and rotation angle
3. WHEN the System initializes the camera, THE System SHALL build the GStreamer pipeline from configuration parameters
4. WHEN camera type is CSI, THE System SHALL use nvarguscamerasrc with configured parameters
5. WHEN camera type is USB, THE System SHALL use v4l2src with configured device path

### Requirement 9: Comprehensive Test Suite

**User Story:** As a developer, I want automated tests for all core functionality, so that I can refactor with confidence and catch regressions early.

#### Acceptance Criteria

1. THE System SHALL provide unit tests for each Model Util covering model loading, inference, and unloading
2. THE System SHALL provide integration tests for the complete Pipeline processing a test video
3. THE System SHALL provide tests for configuration loading with valid, invalid, and missing config files
4. THE System SHALL provide tests for logging system initialization and output formatting
5. THE System SHALL achieve at least 70% code coverage across core modules
6. WHEN tests run, THE System SHALL use mock models and test fixtures to avoid requiring actual TFLite files

### Requirement 10: Performance Profiling Tools

**User Story:** As a developer, I want built-in performance profiling, so that I can identify bottlenecks and optimize the system.

#### Acceptance Criteria

1. THE System SHALL provide a --profile command-line flag for all pipeline modes
2. WHEN profiling is enabled, THE System SHALL log execution time for each pipeline stage
3. WHEN profiling is enabled, THE System SHALL log frame processing rates and queue depths
4. WHEN profiling is enabled, THE System SHALL save detailed profiling data to a file
5. THE System SHALL provide a context manager for timing arbitrary code sections

### Requirement 11: Improved Monitoring System

**User Story:** As a developer, I want a modular monitoring system, so that I can easily add support for new hardware platforms.

#### Acceptance Criteria

1. THE System SHALL provide a base HardwareMonitor class with common metric collection logic
2. THE System SHALL provide platform-specific monitor classes (JetsonMonitor, RaspberryPiMonitor, GenericMonitor)
3. WHEN the System starts monitoring, THE System SHALL auto-detect the hardware platform
4. WHEN the System collects metrics, THE System SHALL use platform-specific APIs when available
5. WHEN platform-specific metrics are unavailable, THE System SHALL fall back to generic metrics without failing

### Requirement 12: Documentation and Type Hints

**User Story:** As a developer, I want comprehensive docstrings and type hints, so that I can understand function contracts without reading implementation details.

#### Acceptance Criteria

1. THE System SHALL provide docstrings for all public functions and classes following Google style
2. THE System SHALL include type hints for all function parameters and return values
3. THE System SHALL document all configuration file options with descriptions and default values
4. THE System SHALL provide a migration guide documenting breaking changes from the original codebase
5. THE System SHALL update the README with new usage examples reflecting the refactored structure

### Requirement 13: Dependency Management

**User Story:** As a developer, I want properly managed dependencies, so that installation is reproducible and straightforward.

#### Acceptance Criteria

1. THE System SHALL provide a requirements.txt file with pinned versions for all dependencies
2. THE System SHALL provide a requirements-dev.txt file for development dependencies (pytest, black, flake8)
3. THE System SHALL provide a requirements-jetson.txt file for Jetson-specific dependencies
4. THE System SHALL document the installation process for each target platform
5. THE System SHALL specify minimum Python version (3.9+)

### Requirement 14: Command-Line Interface Consistency

**User Story:** As a user, I want consistent command-line interfaces across all pipeline modes, so that I can easily switch between offline, real-time, and monitoring modes.

#### Acceptance Criteria

1. THE System SHALL provide consistent argument names across all pipeline scripts (--config, --log-level, --profile)
2. THE System SHALL provide --help output documenting all available options
3. THE System SHALL validate command-line arguments and provide clear error messages for invalid inputs
4. THE System SHALL support environment variables for configuration overrides
5. THE System SHALL provide a --dry-run mode that validates configuration without running the pipeline

### Requirement 15: Graceful Shutdown

**User Story:** As a user, I want the system to shut down cleanly when interrupted, so that no data is lost and resources are properly released.

#### Acceptance Criteria

1. WHEN the user presses Ctrl+C, THE System SHALL catch the interrupt signal
2. WHEN shutting down, THE System SHALL stop all Processing Threads in order
3. WHEN shutting down, THE System SHALL wait for in-progress detections to complete saving
4. WHEN shutting down, THE System SHALL close all open file handles and camera connections
5. WHEN shutting down, THE System SHALL log a summary of processed frames and saved detections
6. THE System SHALL complete shutdown within 10 seconds of receiving the interrupt signal
