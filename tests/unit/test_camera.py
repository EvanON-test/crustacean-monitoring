"""
Unit tests for camera module.

Tests the camera abstraction layer including BaseCamera interface,
GStreamerCamera, OpenCVCamera, and the factory function.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import yaml

from crustacean.camera import (
    BaseCamera,
    GStreamerCamera,
    OpenCVCamera,
    create_camera,
)
from crustacean.utils.config import Config
from crustacean.utils.exceptions import CameraInitError, ConfigurationError


@pytest.fixture
def camera_config_dict():
    """Configuration for camera testing."""
    return {
        'camera': {
            'type': 'csi',
            'width': 1280,
            'height': 720,
            'framerate': 45,
            'rotation': 180,
            'device': '/dev/video0'
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }


@pytest.fixture
def camera_config(temp_dir, camera_config_dict):
    """Create Config object for camera testing."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(camera_config_dict, f)
    return Config.load(str(config_path))


class TestBaseCamera:
    """Test BaseCamera abstract class."""
    
    def test_cannot_instantiate_directly(self, camera_config):
        """Test that BaseCamera cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCamera(camera_config)
    
    def test_get_frame_size(self, camera_config):
        """Test get_frame_size returns configured dimensions."""
        # Create a concrete implementation for testing
        class TestCamera(BaseCamera):
            def open(self): return True
            def read(self): return None
            def release(self): pass
            def is_opened(self): return False
        
        camera = TestCamera(camera_config)
        width, height = camera.get_frame_size()
        
        assert width == 1280
        assert height == 720
    
    def test_get_framerate(self, camera_config):
        """Test get_framerate returns configured framerate."""
        class TestCamera(BaseCamera):
            def open(self): return True
            def read(self): return None
            def release(self): pass
            def is_opened(self): return False
        
        camera = TestCamera(camera_config)
        fps = camera.get_framerate()
        
        assert fps == 45
    
    def test_context_manager_success(self, camera_config):
        """Test context manager opens and releases camera."""
        class TestCamera(BaseCamera):
            def __init__(self, config):
                super().__init__(config)
                self._opened = False
            def open(self):
                self._opened = True
                return True
            def read(self): return None
            def release(self):
                self._opened = False
            def is_opened(self):
                return self._opened
        
        camera = TestCamera(camera_config)
        
        with camera as cam:
            assert cam.is_opened()
        
        assert not camera.is_opened()
    
    def test_context_manager_failure(self, camera_config):
        """Test context manager raises on open failure."""
        class FailingCamera(BaseCamera):
            def open(self): return False
            def read(self): return None
            def release(self): pass
            def is_opened(self): return False
        
        camera = FailingCamera(camera_config)
        
        with pytest.raises(CameraInitError):
            with camera:
                pass
    
    def test_repr(self, camera_config):
        """Test string representation."""
        class TestCamera(BaseCamera):
            def open(self): return True
            def read(self): return None
            def release(self): pass
            def is_opened(self): return False
        
        camera = TestCamera(camera_config)
        repr_str = repr(camera)
        
        assert 'TestCamera' in repr_str
        assert '1280x720' in repr_str
        assert 'closed' in repr_str


class TestGStreamerCamera:
    """Test GStreamerCamera implementation."""
    
    def test_init_builds_pipeline(self, camera_config):
        """Test that initialization builds GStreamer pipeline."""
        camera = GStreamerCamera(camera_config)
        
        assert camera.pipeline is not None
        assert 'nvarguscamerasrc' in camera.pipeline
        assert '1280' in camera.pipeline
        assert '720' in camera.pipeline
        assert '45' in camera.pipeline
    
    def test_pipeline_contains_rotation(self, camera_config):
        """Test that pipeline includes rotation."""
        camera = GStreamerCamera(camera_config)
        
        assert 'rotate-180' in camera.pipeline
    
    def test_rotation_mapping(self, camera_config):
        """Test rotation degree to method mapping."""
        camera = GStreamerCamera(camera_config)
        
        assert camera._get_rotation_method(0) == 'none'
        assert camera._get_rotation_method(90) == 'clockwise'
        assert camera._get_rotation_method(180) == 'rotate-180'
        assert camera._get_rotation_method(270) == 'counterclockwise'
        assert camera._get_rotation_method(45) == 'rotate-180'  # Default
    
    @patch('crustacean.camera.gstreamer_camera.cv2.VideoCapture')
    def test_open_success(self, mock_capture, camera_config):
        """Test successful camera open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        camera = GStreamerCamera(camera_config)
        result = camera.open()
        
        assert result is True
        mock_capture.assert_called_once()
    
    @patch('crustacean.camera.gstreamer_camera.cv2.VideoCapture')
    def test_open_failure_raises_error(self, mock_capture, camera_config):
        """Test that open failure raises CameraInitError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        camera = GStreamerCamera(camera_config)
        
        with pytest.raises(CameraInitError):
            camera.open()
    
    @patch('crustacean.camera.gstreamer_camera.cv2.VideoCapture')
    def test_read_returns_frame(self, mock_capture, camera_config):
        """Test that read returns frame."""
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_capture.return_value = mock_cap
        
        camera = GStreamerCamera(camera_config)
        camera.open()
        frame = camera.read()
        
        assert frame is not None
        assert frame.shape == (720, 1280, 3)
    
    @patch('crustacean.camera.gstreamer_camera.cv2.VideoCapture')
    def test_read_returns_none_on_failure(self, mock_capture, camera_config):
        """Test that read returns None on failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_capture.return_value = mock_cap
        
        camera = GStreamerCamera(camera_config)
        camera.open()
        frame = camera.read()
        
        assert frame is None
    
    @patch('crustacean.camera.gstreamer_camera.cv2.VideoCapture')
    def test_release(self, mock_capture, camera_config):
        """Test camera release."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        camera = GStreamerCamera(camera_config)
        camera.open()
        camera.release()
        
        mock_cap.release.assert_called_once()
        assert camera.capture is None
    
    def test_release_without_open(self, camera_config):
        """Test release without opening doesn't raise."""
        camera = GStreamerCamera(camera_config)
        camera.release()  # Should not raise
    
    def test_is_opened_false_initially(self, camera_config):
        """Test is_opened returns False initially."""
        camera = GStreamerCamera(camera_config)
        
        assert not camera.is_opened()
    
    def test_get_pipeline(self, camera_config):
        """Test get_pipeline returns pipeline string."""
        camera = GStreamerCamera(camera_config)
        
        assert camera.get_pipeline() == camera.pipeline


class TestOpenCVCamera:
    """Test OpenCVCamera implementation."""
    
    def test_init_sets_device(self, camera_config):
        """Test that initialization sets device."""
        camera = OpenCVCamera(camera_config)
        
        assert camera.device == '/dev/video0'
    
    def test_device_string_to_int(self, temp_dir):
        """Test that numeric string device is converted to int."""
        config_dict = {
            'camera': {'device': '0'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = OpenCVCamera(config)
        
        assert camera.device == 0
    
    @patch('crustacean.camera.opencv_camera.cv2.VideoCapture')
    def test_open_success(self, mock_capture, camera_config):
        """Test successful camera open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_config)
        result = camera.open()
        
        assert result is True
    
    @patch('crustacean.camera.opencv_camera.cv2.VideoCapture')
    def test_open_configures_capture(self, mock_capture, camera_config):
        """Test that open configures capture properties."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_config)
        camera.open()
        
        # Verify properties were set
        mock_cap.set.assert_any_call(3, 1280)  # CAP_PROP_FRAME_WIDTH
        mock_cap.set.assert_any_call(4, 720)   # CAP_PROP_FRAME_HEIGHT
    
    @patch('crustacean.camera.opencv_camera.cv2.VideoCapture')
    def test_open_failure_raises_error(self, mock_capture, camera_config):
        """Test that open failure raises CameraInitError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_config)
        
        with pytest.raises(CameraInitError):
            camera.open()
    
    @patch('crustacean.camera.opencv_camera.cv2.VideoCapture')
    def test_read_returns_frame(self, mock_capture, camera_config):
        """Test that read returns frame."""
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.get.return_value = 30.0
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_config)
        camera.open()
        frame = camera.read()
        
        assert frame is not None
        assert frame.shape == (720, 1280, 3)
    
    @patch('crustacean.camera.opencv_camera.cv2.VideoCapture')
    def test_release(self, mock_capture, camera_config):
        """Test camera release."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_config)
        camera.open()
        camera.release()
        
        mock_cap.release.assert_called_once()
        assert camera.capture is None
    
    def test_get_device(self, camera_config):
        """Test get_device returns device."""
        camera = OpenCVCamera(camera_config)
        
        assert camera.get_device() == '/dev/video0'


class TestCameraFactory:
    """Test create_camera factory function."""
    
    def test_create_csi_camera(self, camera_config):
        """Test creating CSI camera."""
        camera = create_camera(camera_config)
        
        assert isinstance(camera, GStreamerCamera)
    
    def test_create_usb_camera(self, temp_dir):
        """Test creating USB camera."""
        config_dict = {
            'camera': {'type': 'usb'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = create_camera(config)
        
        assert isinstance(camera, OpenCVCamera)
    
    def test_create_gstreamer_alias(self, temp_dir):
        """Test 'gstreamer' alias creates GStreamerCamera."""
        config_dict = {
            'camera': {'type': 'gstreamer'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = create_camera(config)
        
        assert isinstance(camera, GStreamerCamera)
    
    def test_create_opencv_alias(self, temp_dir):
        """Test 'opencv' alias creates OpenCVCamera."""
        config_dict = {
            'camera': {'type': 'opencv'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = create_camera(config)
        
        assert isinstance(camera, OpenCVCamera)
    
    def test_invalid_camera_type_raises_error(self, temp_dir):
        """Test that invalid camera type raises ConfigurationError."""
        config_dict = {
            'camera': {'type': 'invalid'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        with pytest.raises(ConfigurationError) as exc_info:
            create_camera(config)
        
        assert 'invalid' in str(exc_info.value)
    
    def test_case_insensitive_type(self, temp_dir):
        """Test that camera type is case insensitive."""
        config_dict = {
            'camera': {'type': 'CSI'},
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = create_camera(config)
        
        assert isinstance(camera, GStreamerCamera)
    
    def test_default_camera_type(self, temp_dir):
        """Test default camera type is CSI."""
        config_dict = {
            'camera': {},  # No type specified
            'logging': {'level': 'INFO'}
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        camera = create_camera(config)
        
        assert isinstance(camera, GStreamerCamera)
