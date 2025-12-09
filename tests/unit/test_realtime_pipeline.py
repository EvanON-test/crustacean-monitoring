"""
Unit tests for RealtimePipeline class.

Tests the RealtimePipeline including initialization, motion detection,
frame collection, and shutdown functionality.
"""

import pytest
import queue
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import yaml

from crustacean.core.realtime_pipeline import RealtimePipeline
from crustacean.utils.config import Config
from crustacean.utils.exceptions import CameraInitError


@pytest.fixture
def realtime_config_dict():
    """Configuration for RealtimePipeline testing."""
    return {
        'models': {
            'binary_classifier': {'path': 'test/bc.tflite'},
            'frame_selector': {
                'top_model_path': 'test/top.tflite',
                'bottom_model_path': 'test/bottom.tflite'
            },
            'object_detector': {
                'path': 'test/od.tflite',
                'confidence_threshold': 0.75
            },
            'keypoint_detector': {'path': 'test/kd.tflite'}
        },
        'camera': {
            'type': 'usb',
            'width': 640,
            'height': 480
        },
        'realtime': {
            'motion_detection_threshold': 15,
            'detection_cooldown': 3,
            'frames_to_collect': 30,
            'process_interval': 30,
            'max_save_threads': 2
        },
        'output': {
            'detections_dir': './test_detections'
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }


@pytest.fixture
def realtime_config(temp_dir, realtime_config_dict):
    """Create Config object for RealtimePipeline testing."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(realtime_config_dict, f)
    return Config.load(str(config_path))


class TestRealtimePipelineInitialization:
    """Test RealtimePipeline initialization."""
    
    def test_init_sets_display_mode(self, realtime_config):
        """Test that initialization sets display mode."""
        pipeline = RealtimePipeline(realtime_config, display_mode=True)
        assert pipeline.display_mode is True
        
        pipeline2 = RealtimePipeline(realtime_config, display_mode=False)
        assert pipeline2.display_mode is False
    
    def test_init_sets_config_values(self, realtime_config):
        """Test that initialization reads config values."""
        pipeline = RealtimePipeline(realtime_config)
        
        assert pipeline.motion_threshold == 15
        assert pipeline.detection_cooldown == 3
        assert pipeline.frames_to_collect == 30
        assert pipeline.process_interval == 30
    
    def test_init_empty_state(self, realtime_config):
        """Test that initialization starts with empty state."""
        pipeline = RealtimePipeline(realtime_config)
        
        assert pipeline.camera is None
        assert pipeline.threads == {}
        assert pipeline.queues == {}
        assert pipeline.executor is None
        assert pipeline.collecting is False
        assert pipeline.collected_frames == []
        assert pipeline.frame_counter == 0
        assert pipeline.detection_count == 0


class TestMotionDetection:
    """Test motion detection functionality."""
    
    def test_detect_motion_first_frame(self, realtime_config):
        """Test that first frame returns False (no previous frame)."""
        pipeline = RealtimePipeline(realtime_config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline._detect_motion(frame)
        
        assert result is False
        assert pipeline.previous_frame is not None
    
    def test_detect_motion_no_change(self, realtime_config):
        """Test that identical frames return False."""
        pipeline = RealtimePipeline(realtime_config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._detect_motion(frame)  # Initialize
        result = pipeline._detect_motion(frame)  # Same frame
        
        assert result is False
    
    def test_detect_motion_with_change(self, realtime_config):
        """Test that significant change returns True."""
        pipeline = RealtimePipeline(realtime_config)
        
        # First frame - all black
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._detect_motion(frame1)
        
        # Second frame - significant white area
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[:240, :320] = 255  # 25% of frame is white
        
        result = pipeline._detect_motion(frame2)
        
        assert result is True


class TestFrameCollection:
    """Test frame collection functionality."""
    
    def test_should_process_frame_at_interval(self, realtime_config):
        """Test frame processing at correct intervals."""
        pipeline = RealtimePipeline(realtime_config)
        pipeline.process_interval = 30
        
        assert pipeline._should_process_frame(0) is True
        assert pipeline._should_process_frame(30) is True
        assert pipeline._should_process_frame(60) is True
        assert pipeline._should_process_frame(15) is False
        assert pipeline._should_process_frame(31) is False
    
    def test_collect_frame_adds_to_list(self, realtime_config):
        """Test that collect_frame adds frames to list."""
        pipeline = RealtimePipeline(realtime_config)
        pipeline.collecting = True
        pipeline.frames_to_collect = 5
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._collect_frame(frame)
        
        assert len(pipeline.collected_frames) == 1
    
    def test_collect_frame_triggers_submit(self, realtime_config):
        """Test that collection triggers submit when full."""
        pipeline = RealtimePipeline(realtime_config)
        pipeline.collecting = True
        pipeline.frames_to_collect = 3
        pipeline.queues['analysis'] = queue.Queue(maxsize=1)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Collect frames
        pipeline._collect_frame(frame)
        pipeline._collect_frame(frame)
        
        assert pipeline.collecting is True
        assert len(pipeline.collected_frames) == 2
        
        # Third frame should trigger submit
        pipeline._collect_frame(frame)
        
        assert pipeline.collecting is False
        assert len(pipeline.collected_frames) == 0


class TestQueueInitialization:
    """Test queue initialization."""
    
    def test_initialize_queues(self, realtime_config):
        """Test that queues are initialized correctly."""
        pipeline = RealtimePipeline(realtime_config)
        pipeline._initialize_queues()
        
        assert 'analysis' in pipeline.queues
        assert 'detection' in pipeline.queues
        assert 'results' in pipeline.queues
        
        assert isinstance(pipeline.queues['analysis'], queue.Queue)
        assert isinstance(pipeline.queues['detection'], queue.Queue)
        assert isinstance(pipeline.queues['results'], queue.Queue)


class TestShutdown:
    """Test shutdown functionality."""
    
    def test_stop_threads_empty(self, realtime_config):
        """Test stopping threads when none exist."""
        pipeline = RealtimePipeline(realtime_config)
        
        # Should not raise
        pipeline._stop_threads()
        
        assert pipeline.threads == {}
    
    def test_shutdown_executor_none(self, realtime_config):
        """Test shutdown executor when None."""
        pipeline = RealtimePipeline(realtime_config)
        
        # Should not raise
        pipeline._shutdown_executor()
        
        assert pipeline.executor is None
    
    def test_release_camera_none(self, realtime_config):
        """Test release camera when None."""
        pipeline = RealtimePipeline(realtime_config)
        
        # Should not raise
        pipeline._release_camera()
        
        assert pipeline.camera is None
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    def test_release_camera(self, mock_create_camera, realtime_config):
        """Test camera release."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_create_camera.return_value = mock_camera
        
        pipeline = RealtimePipeline(realtime_config)
        pipeline._initialize_camera()
        pipeline._release_camera()
        
        mock_camera.release.assert_called_once()
        assert pipeline.camera is None


class TestCameraInitialization:
    """Test camera initialization."""
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    def test_initialize_camera_success(self, mock_create_camera, realtime_config):
        """Test successful camera initialization."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_create_camera.return_value = mock_camera
        
        pipeline = RealtimePipeline(realtime_config)
        pipeline._initialize_camera()
        
        assert pipeline.camera is mock_camera
        mock_camera.open.assert_called_once()
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    def test_initialize_camera_failure(self, mock_create_camera, realtime_config):
        """Test camera initialization failure raises error."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = False
        mock_create_camera.return_value = mock_camera
        
        pipeline = RealtimePipeline(realtime_config)
        
        with pytest.raises(CameraInitError):
            pipeline._initialize_camera()


class TestDisplayMode:
    """Test display mode functionality."""
    
    def test_draw_overlay(self, realtime_config):
        """Test overlay drawing doesn't crash."""
        pipeline = RealtimePipeline(realtime_config)
        pipeline.frame_counter = 100
        pipeline.detection_count = 5
        pipeline.start_time = time.time() - 60
        pipeline.latest_confidence = 0.85
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise
        pipeline._draw_overlay(frame)
        
        # Frame should be modified (not all zeros)
        assert frame.sum() > 0


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_cleanup(self, realtime_config):
        """Test that context manager calls cleanup."""
        pipeline = RealtimePipeline(realtime_config)
        
        with pipeline:
            pass
        
        assert pipeline.models == {}


class TestRepr:
    """Test string representation."""
    
    def test_repr(self, realtime_config):
        """Test repr output."""
        pipeline = RealtimePipeline(realtime_config)
        
        repr_str = repr(pipeline)
        
        assert 'RealtimePipeline' in repr_str
