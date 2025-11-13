"""
Unit tests for custom exception hierarchy.

Tests all custom exceptions, inheritance, error messages,
and details handling.
"""

import pytest
from crustacean.utils.exceptions import (
    CrustaceanError,
    ConfigurationError,
    ModelLoadError,
    ModelNotLoadedError,
    CameraInitError,
    InferenceError,
    ThreadError,
    VideoProcessingError,
    DetectionSaveError,
    MonitoringError
)


class TestBaseException:
    """Test the base CrustaceanError exception."""
    
    def test_base_exception_with_message(self):
        """Test creating base exception with message only."""
        error = CrustaceanError("Test error")
        assert error.message == "Test error"
        assert error.details == {}
        assert str(error) == "Test error"
    
    def test_base_exception_with_details(self):
        """Test creating base exception with message and details."""
        error = CrustaceanError("Test error", details={'key': 'value', 'count': 42})
        assert error.message == "Test error"
        assert error.details == {'key': 'value', 'count': 42}
        assert 'key=value' in str(error)
        assert 'count=42' in str(error)
    
    def test_base_exception_is_exception(self):
        """Test that CrustaceanError is an Exception."""
        error = CrustaceanError("Test")
        assert isinstance(error, Exception)
    
    def test_base_exception_can_be_raised(self):
        """Test that base exception can be raised and caught."""
        with pytest.raises(CrustaceanError) as exc_info:
            raise CrustaceanError("Test error")
        
        assert exc_info.value.message == "Test error"


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error_inherits_base(self):
        """Test that ConfigurationError inherits from CrustaceanError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, CrustaceanError)
        assert isinstance(error, Exception)
    
    def test_configuration_error_with_details(self):
        """Test ConfigurationError with details."""
        error = ConfigurationError(
            "Missing configuration",
            details={'key': 'models.binary_classifier.path'}
        )
        assert error.message == "Missing configuration"
        assert error.details['key'] == 'models.binary_classifier.path'
    
    def test_configuration_error_can_be_caught_specifically(self):
        """Test catching ConfigurationError specifically."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error")
    
    def test_configuration_error_can_be_caught_as_base(self):
        """Test catching ConfigurationError as CrustaceanError."""
        with pytest.raises(CrustaceanError):
            raise ConfigurationError("Config error")


class TestModelErrors:
    """Test model-related exceptions."""
    
    def test_model_load_error(self):
        """Test ModelLoadError exception."""
        error = ModelLoadError(
            "Failed to load model",
            details={'path': 'model.tflite', 'error': 'File not found'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.message == "Failed to load model"
        assert error.details['path'] == 'model.tflite'
    
    def test_model_not_loaded_error(self):
        """Test ModelNotLoadedError exception."""
        error = ModelNotLoadedError(
            "Model not loaded",
            details={'model': 'BinaryClassifier'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.message == "Model not loaded"
    
    def test_inference_error(self):
        """Test InferenceError exception."""
        error = InferenceError(
            "Inference failed",
            details={'frame': 42, 'model': 'ObjectDetector'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.details['frame'] == 42


class TestCameraError:
    """Test camera-related exceptions."""
    
    def test_camera_init_error(self):
        """Test CameraInitError exception."""
        error = CameraInitError(
            "Failed to open camera",
            details={'device': '/dev/video0', 'type': 'csi'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.message == "Failed to open camera"
        assert error.details['device'] == '/dev/video0'
    
    def test_camera_init_error_without_details(self):
        """Test CameraInitError without details."""
        error = CameraInitError("Camera not found")
        assert error.message == "Camera not found"
        assert error.details == {}


class TestThreadError:
    """Test thread-related exceptions."""
    
    def test_thread_error(self):
        """Test ThreadError exception."""
        error = ThreadError(
            "Thread failed",
            details={'thread': 'AnalysisThread', 'error': 'Timeout'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.details['thread'] == 'AnalysisThread'


class TestVideoProcessingError:
    """Test video processing exceptions."""
    
    def test_video_processing_error(self):
        """Test VideoProcessingError exception."""
        error = VideoProcessingError(
            "Failed to read video",
            details={'path': 'video.mp4', 'frame': 100}
        )
        assert isinstance(error, CrustaceanError)
        assert error.details['path'] == 'video.mp4'
        assert error.details['frame'] == 100


class TestDetectionSaveError:
    """Test detection saving exceptions."""
    
    def test_detection_save_error(self):
        """Test DetectionSaveError exception."""
        error = DetectionSaveError(
            "Failed to save detection",
            details={'directory': 'realtime_frames/', 'error': 'Permission denied'}
        )
        assert isinstance(error, CrustaceanError)
        assert 'directory' in error.details


class TestMonitoringError:
    """Test monitoring exceptions."""
    
    def test_monitoring_error(self):
        """Test MonitoringError exception."""
        error = MonitoringError(
            "Failed to collect metrics",
            details={'platform': 'jetson', 'metric': 'gpu_temp'}
        )
        assert isinstance(error, CrustaceanError)
        assert error.details['platform'] == 'jetson'


class TestExceptionHierarchy:
    """Test exception hierarchy and inheritance."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from CrustaceanError."""
        exceptions = [
            ConfigurationError,
            ModelLoadError,
            ModelNotLoadedError,
            CameraInitError,
            InferenceError,
            ThreadError,
            VideoProcessingError,
            DetectionSaveError,
            MonitoringError
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, CrustaceanError)
            assert issubclass(exc_class, Exception)
    
    def test_catch_all_with_base_exception(self):
        """Test catching any custom exception with CrustaceanError."""
        exceptions_to_test = [
            ConfigurationError("test"),
            ModelLoadError("test"),
            CameraInitError("test"),
            InferenceError("test"),
            ThreadError("test")
        ]
        
        for exc in exceptions_to_test:
            with pytest.raises(CrustaceanError):
                raise exc
    
    def test_specific_exception_not_caught_by_sibling(self):
        """Test that specific exceptions are not caught by sibling exceptions."""
        with pytest.raises(ModelLoadError):
            try:
                raise ModelLoadError("test")
            except CameraInitError:
                pytest.fail("Should not catch ModelLoadError as CameraInitError")


class TestExceptionMessages:
    """Test exception message formatting."""
    
    def test_message_without_details(self):
        """Test exception message without details."""
        error = CrustaceanError("Simple error")
        assert str(error) == "Simple error"
    
    def test_message_with_single_detail(self):
        """Test exception message with single detail."""
        error = CrustaceanError("Error", details={'key': 'value'})
        message = str(error)
        assert "Error" in message
        assert "key=value" in message
    
    def test_message_with_multiple_details(self):
        """Test exception message with multiple details."""
        error = CrustaceanError(
            "Error",
            details={'key1': 'value1', 'key2': 'value2'}
        )
        message = str(error)
        assert "Error" in message
        assert "key1=value1" in message
        assert "key2=value2" in message
    
    def test_message_with_numeric_details(self):
        """Test exception message with numeric details."""
        error = CrustaceanError(
            "Error",
            details={'count': 42, 'rate': 3.14}
        )
        message = str(error)
        assert "count=42" in message
        assert "rate=3.14" in message


class TestExceptionUsage:
    """Test practical exception usage patterns."""
    
    def test_exception_in_try_except(self):
        """Test using exceptions in try-except blocks."""
        def risky_operation():
            raise ModelLoadError("Model not found", details={'path': 'model.tflite'})
        
        with pytest.raises(ModelLoadError) as exc_info:
            risky_operation()
        
        assert exc_info.value.message == "Model not found"
        assert exc_info.value.details['path'] == 'model.tflite'
    
    def test_exception_chaining(self):
        """Test exception chaining with from clause."""
        original_error = ValueError("Original error")
        
        with pytest.raises(ModelLoadError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise ModelLoadError("Failed to load", details={'error': str(e)}) from e
        
        assert exc_info.value.details['error'] == "Original error"
    
    def test_exception_with_empty_details(self):
        """Test exception with explicitly empty details."""
        error = CrustaceanError("Error", details={})
        assert error.details == {}
        assert str(error) == "Error"
    
    def test_exception_details_are_mutable(self):
        """Test that exception details can be accessed and modified."""
        error = CrustaceanError("Error", details={'key': 'value'})
        error.details['new_key'] = 'new_value'
        assert error.details['new_key'] == 'new_value'
