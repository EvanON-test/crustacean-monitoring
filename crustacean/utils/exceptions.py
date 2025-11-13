"""
Custom exception hierarchy for Crustacean Monitoring System.

This module defines a hierarchy of exceptions for different error conditions
throughout the system. All exceptions inherit from CrustaceanError, allowing
for both specific and general exception handling.

Example:
    >>> try:
    ...     model.load()
    ... except ModelLoadError as e:
    ...     logger.error(f"Failed to load model: {e}")
    ... except CrustaceanError as e:
    ...     logger.error(f"General error: {e}")
"""


class CrustaceanError(Exception):
    """
    Base exception for all Crustacean Monitoring System errors.
    
    All custom exceptions in the system inherit from this class,
    allowing for catching all system-specific errors with a single
    except clause.
    
    Attributes:
        message: Error message describing what went wrong
        details: Optional dictionary with additional error context
    """
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(CrustaceanError):
    """
    Configuration file or validation errors.
    
    Raised when:
    - Configuration file is missing or malformed
    - Required configuration values are missing
    - Configuration values are invalid
    - Schema validation fails
    
    Example:
        >>> raise ConfigurationError(
        ...     "Missing required configuration",
        ...     details={'key': 'models.binary_classifier.path'}
        ... )
    """
    pass


class ModelLoadError(CrustaceanError):
    """
    Model loading failures.
    
    Raised when:
    - TFLite model file not found
    - Model file is corrupted
    - Model architecture incompatible
    - Insufficient memory to load model
    
    Example:
        >>> raise ModelLoadError(
        ...     "Failed to load TFLite model",
        ...     details={'path': 'model.tflite', 'error': str(e)}
        ... )
    """
    pass


class ModelNotLoadedError(CrustaceanError):
    """
    Attempting inference without loaded model.
    
    Raised when:
    - predict() called before load()
    - Model was unloaded but inference attempted
    - Model loading failed but code continued
    
    Example:
        >>> raise ModelNotLoadedError(
        ...     "Model must be loaded before inference",
        ...     details={'model': 'BinaryClassifier'}
        ... )
    """
    pass


class CameraInitError(CrustaceanError):
    """
    Camera initialization failures.
    
    Raised when:
    - Camera device not found
    - GStreamer pipeline fails to initialize
    - Camera permissions denied
    - Camera already in use by another process
    
    Example:
        >>> raise CameraInitError(
        ...     "Failed to open camera",
        ...     details={'device': '/dev/video0', 'type': 'csi'}
        ... )
    """
    pass


class InferenceError(CrustaceanError):
    """
    Model inference failures.
    
    Raised when:
    - Input data shape mismatch
    - Inference computation fails
    - Output tensor extraction fails
    - Numerical errors during inference
    
    Example:
        >>> raise InferenceError(
        ...     "Inference failed on frame",
        ...     details={'frame': 42, 'model': 'ObjectDetector'}
        ... )
    """
    pass


class ThreadError(CrustaceanError):
    """
    Thread management errors.
    
    Raised when:
    - Thread fails to start
    - Thread encounters unhandled exception
    - Thread join timeout exceeded
    - Queue operations fail
    
    Example:
        >>> raise ThreadError(
        ...     "Analysis thread failed",
        ...     details={'thread': 'AnalysisThread', 'error': str(e)}
        ... )
    """
    pass


class VideoProcessingError(CrustaceanError):
    """
    Video file processing errors.
    
    Raised when:
    - Video file not found
    - Video file corrupted or unreadable
    - Video codec not supported
    - Frame extraction fails
    
    Example:
        >>> raise VideoProcessingError(
        ...     "Failed to read video file",
        ...     details={'path': 'video.mp4', 'frame': 100}
        ... )
    """
    pass


class DetectionSaveError(CrustaceanError):
    """
    Detection result saving errors.
    
    Raised when:
    - Cannot create output directory
    - Disk space insufficient
    - File write permissions denied
    - CSV writing fails
    
    Example:
        >>> raise DetectionSaveError(
        ...     "Failed to save detection",
        ...     details={'directory': 'realtime_frames/', 'error': str(e)}
        ... )
    """
    pass


class MonitoringError(CrustaceanError):
    """
    Hardware monitoring errors.
    
    Raised when:
    - Hardware metrics unavailable
    - Monitoring thread fails
    - CSV writing fails
    - Platform detection fails
    
    Example:
        >>> raise MonitoringError(
        ...     "Failed to collect hardware metrics",
        ...     details={'platform': 'jetson', 'metric': 'gpu_temp'}
        ... )
    """
    pass
