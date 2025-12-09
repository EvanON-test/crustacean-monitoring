"""
Unit tests for thread classes.

Tests the AnalysisThread, DetectionThread, and save_detection function.
"""

import pytest
import queue
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import yaml

from crustacean.threads import (
    AnalysisThread,
    DetectionThread,
    DetectionResult,
    save_detection,
)
from crustacean.utils.config import Config


@pytest.fixture
def thread_config_dict():
    """Configuration for thread testing."""
    return {
        'models': {
            'object_detector': {
                'confidence_threshold': 0.75
            },
            'keypoint_detector': {
                'path': 'test/kd.tflite'
            }
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
def thread_config(temp_dir, thread_config_dict):
    """Create Config object for thread testing."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(thread_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def mock_bc_model():
    """Mock Binary Classifier model."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 1, 1, 0])
    return model


@pytest.fixture
def mock_fs_model():
    """Mock Frame Selector model."""
    model = MagicMock()
    model.predict.return_value = [[1], [2]]  # top and bottom indices
    return model


@pytest.fixture
def mock_od_model():
    """Mock Object Detector model."""
    model = MagicMock()
    roi = np.zeros((100, 100, 3), dtype=np.uint8)
    model.predict.return_value = (roi, 0.85, 0)
    return model


@pytest.fixture
def mock_kd_model():
    """Mock Keypoint Detector model."""
    model = MagicMock()
    model.predict.return_value = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]])
    return model


@pytest.fixture
def sample_frames():
    """Create sample frames for testing."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]


class TestAnalysisThread:
    """Test AnalysisThread class."""
    
    def test_init(self, mock_bc_model, mock_fs_model):
        """Test thread initialization."""
        analysis_queue = queue.Queue()
        detection_queue = queue.Queue()
        
        thread = AnalysisThread(
            analysis_queue, detection_queue, mock_bc_model, mock_fs_model
        )
        
        assert thread.analysis_queue is analysis_queue
        assert thread.detection_queue is detection_queue
        assert thread.bc is mock_bc_model
        assert thread.fs is mock_fs_model
        assert thread.running is True
    
    def test_stop(self, mock_bc_model, mock_fs_model):
        """Test stop method sets running to False."""
        thread = AnalysisThread(
            queue.Queue(), queue.Queue(), mock_bc_model, mock_fs_model
        )
        
        thread.stop()
        
        assert thread.running is False
    
    def test_thread_name(self, mock_bc_model, mock_fs_model):
        """Test thread has correct name."""
        thread = AnalysisThread(
            queue.Queue(), queue.Queue(), mock_bc_model, mock_fs_model
        )
        
        assert thread.name == "AnalysisThread"
    
    def test_daemon_thread(self, mock_bc_model, mock_fs_model):
        """Test thread is daemon."""
        thread = AnalysisThread(
            queue.Queue(), queue.Queue(), mock_bc_model, mock_fs_model
        )
        
        assert thread.daemon is True
    
    @patch('crustacean.threads.analysis_thread.cv2.VideoCapture')
    @patch('crustacean.threads.analysis_thread.cv2.VideoWriter')
    def test_process_frames_no_detection(
        self, mock_writer, mock_capture, mock_bc_model, mock_fs_model, sample_frames
    ):
        """Test processing when BC detects nothing."""
        # BC returns all zeros (no detection)
        mock_bc_model.predict.return_value = np.array([0, 0, 0, 0, 0])
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        analysis_queue = queue.Queue()
        detection_queue = queue.Queue()
        
        thread = AnalysisThread(
            analysis_queue, detection_queue, mock_bc_model, mock_fs_model
        )
        
        # Process frames directly
        thread._process_frames(sample_frames, 0)
        
        # Detection queue should be empty (no positive detection)
        assert detection_queue.empty()


class TestDetectionThread:
    """Test DetectionThread class."""
    
    def test_init(self, mock_od_model, thread_config):
        """Test thread initialization."""
        frame_queue = queue.Queue()
        result_queue = queue.Queue()
        
        thread = DetectionThread(
            frame_queue, result_queue, mock_od_model, thread_config
        )
        
        assert thread.frame_queue is frame_queue
        assert thread.result_queue is result_queue
        assert thread.od is mock_od_model
        assert thread.running is True
        assert thread.confidence_threshold == 0.75
    
    def test_stop(self, mock_od_model):
        """Test stop method sets running to False."""
        thread = DetectionThread(
            queue.Queue(), queue.Queue(), mock_od_model
        )
        
        thread.stop()
        
        assert thread.running is False
    
    def test_thread_name(self, mock_od_model):
        """Test thread has correct name."""
        thread = DetectionThread(
            queue.Queue(), queue.Queue(), mock_od_model
        )
        
        assert thread.name == "DetectionThread"
    
    def test_daemon_thread(self, mock_od_model):
        """Test thread is daemon."""
        thread = DetectionThread(
            queue.Queue(), queue.Queue(), mock_od_model
        )
        
        assert thread.daemon is True
    
    def test_process_frame(self, mock_od_model):
        """Test processing a single frame."""
        frame_queue = queue.Queue()
        result_queue = queue.Queue()
        
        thread = DetectionThread(frame_queue, result_queue, mock_od_model)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        thread._process_frame(frame, 100)
        
        # Check result was queued
        assert not result_queue.empty()
        result = result_queue.get()
        
        assert isinstance(result, DetectionResult)
        assert result.frame_number == 100
        assert result.confidence == 0.85
        assert result.class_index == 0


class TestDetectionResult:
    """Test DetectionResult class."""
    
    def test_init(self):
        """Test result initialization."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = DetectionResult(
            frame=frame,
            roi=roi,
            confidence=0.85,
            class_index=0,
            frame_number=100
        )
        
        assert result.confidence == 0.85
        assert result.class_index == 0
        assert result.frame_number == 100
    
    def test_is_high_confidence_true(self):
        """Test is_high_confidence returns True above threshold."""
        result = DetectionResult(
            frame=np.zeros((10, 10, 3)),
            roi=np.zeros((10, 10, 3)),
            confidence=0.85,
            class_index=0,
            frame_number=1
        )
        
        assert result.is_high_confidence(0.75) is True
    
    def test_is_high_confidence_false(self):
        """Test is_high_confidence returns False below threshold."""
        result = DetectionResult(
            frame=np.zeros((10, 10, 3)),
            roi=np.zeros((10, 10, 3)),
            confidence=0.5,
            class_index=0,
            frame_number=1
        )
        
        assert result.is_high_confidence(0.75) is False
    
    def test_repr(self):
        """Test string representation."""
        result = DetectionResult(
            frame=np.zeros((10, 10, 3)),
            roi=np.zeros((10, 10, 3)),
            confidence=0.85,
            class_index=0,
            frame_number=100
        )
        
        repr_str = repr(result)
        
        assert 'DetectionResult' in repr_str
        assert '100' in repr_str
        assert '0.85' in repr_str


class TestSaveDetection:
    """Test save_detection function."""
    
    @patch('crustacean.threads.save_thread.cv2.imwrite')
    def test_save_detection_creates_directory(
        self, mock_imwrite, thread_config, temp_dir, mock_kd_model
    ):
        """Test that save_detection creates output directory."""
        mock_imwrite.return_value = True
        
        # Update config to use temp directory
        thread_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = save_detection(
            frame, roi, 0.85, 100, thread_config, mock_kd_model
        )
        
        assert result is not None
        assert Path(result).exists()
    
    @patch('crustacean.threads.save_thread.cv2.imwrite')
    def test_save_detection_saves_frame(
        self, mock_imwrite, thread_config, temp_dir, mock_kd_model
    ):
        """Test that save_detection saves frame image."""
        mock_imwrite.return_value = True
        
        thread_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        save_detection(frame, roi, 0.85, 100, thread_config, mock_kd_model)
        
        # Verify imwrite was called
        mock_imwrite.assert_called()
    
    @patch('crustacean.threads.save_thread.cv2.imwrite')
    def test_save_detection_saves_keypoints(
        self, mock_imwrite, thread_config, temp_dir, mock_kd_model
    ):
        """Test that save_detection saves keypoints CSV."""
        mock_imwrite.return_value = True
        
        thread_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result_path = save_detection(
            frame, roi, 0.85, 100, thread_config, mock_kd_model
        )
        
        # Check for CSV file
        result_dir = Path(result_path)
        csv_files = list(result_dir.glob('*_keypoints.csv'))
        
        assert len(csv_files) == 1
    
    @patch('crustacean.threads.save_thread.cv2.imwrite')
    def test_save_detection_saves_metadata(
        self, mock_imwrite, thread_config, temp_dir, mock_kd_model
    ):
        """Test that save_detection saves metadata file."""
        mock_imwrite.return_value = True
        
        thread_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result_path = save_detection(
            frame, roi, 0.85, 100, thread_config, mock_kd_model
        )
        
        # Check for metadata file
        result_dir = Path(result_path)
        metadata_files = list(result_dir.glob('*_metadata.txt'))
        
        assert len(metadata_files) == 1
    
    @patch('crustacean.threads.save_thread.cv2.imwrite')
    def test_save_detection_returns_none_on_failure(
        self, mock_imwrite, thread_config, temp_dir
    ):
        """Test that save_detection returns None on frame save failure."""
        mock_imwrite.return_value = False
        
        thread_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = save_detection(frame, roi, 0.85, 100, thread_config)
        
        assert result is None


class TestThreadIntegration:
    """Integration tests for thread coordination."""
    
    def test_analysis_to_detection_queue(self, mock_bc_model, mock_fs_model, mock_od_model):
        """Test that analysis thread can send to detection thread."""
        analysis_queue = queue.Queue(maxsize=1)
        detection_queue = queue.Queue(maxsize=1)
        result_queue = queue.Queue(maxsize=1)
        
        # Create threads (but don't start them)
        analysis_thread = AnalysisThread(
            analysis_queue, detection_queue, mock_bc_model, mock_fs_model
        )
        detection_thread = DetectionThread(
            detection_queue, result_queue, mock_od_model
        )
        
        # Verify queues are connected
        assert analysis_thread.detection_queue is detection_thread.frame_queue
