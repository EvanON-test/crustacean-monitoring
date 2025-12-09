"""
Integration tests for RealtimePipeline.

Tests the complete real-time pipeline processing flow including:
- Initialization with mocked camera and models
- Motion detection triggering frame collection
- Thread coordination between analysis and detection
- Graceful shutdown
"""

import pytest
import queue
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import yaml

from crustacean.core.realtime_pipeline import RealtimePipeline
from crustacean.threads import DetectionResult
from crustacean.utils.config import Config


@pytest.fixture
def integration_config_dict():
    """Configuration for integration testing."""
    return {
        'models': {
            'binary_classifier': {
                'path': 'test/bc_model.tflite',
                'input_width': 320,
                'input_height': 180
            },
            'frame_selector': {
                'top_model_path': 'test/top_model.tflite',
                'bottom_model_path': 'test/bottom_model.tflite'
            },
            'object_detector': {
                'path': 'test/od_model.tflite',
                'input_size': 640,
                'confidence_threshold': 0.75
            },
            'keypoint_detector': {
                'path': 'test/kd_model.tflite',
                'num_keypoints': 7
            }
        },
        'camera': {
            'type': 'usb',
            'width': 640,
            'height': 480,
            'device': '0'
        },
        'realtime': {
            'motion_detection_threshold': 15,
            'detection_cooldown': 1,
            'frames_to_collect': 5,
            'process_interval': 1,
            'max_save_threads': 2
        },
        'output': {
            'detections_dir': './test_detections'
        },
        'logging': {
            'level': 'DEBUG',
            'console': True
        }
    }


@pytest.fixture
def integration_config(temp_dir, integration_config_dict):
    """Create Config object for integration testing."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(integration_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def mock_models():
    """Create mock model instances."""
    models = {}
    
    # Mock Binary Classifier
    bc = MagicMock()
    bc.is_loaded.return_value = True
    bc.predict.return_value = np.array([0, 1, 1, 1, 0])
    models['bc'] = bc
    
    # Mock Frame Selector
    fs = MagicMock()
    fs.is_loaded.return_value = True
    fs.predict.return_value = [[1], [2]]
    models['fs'] = fs
    
    # Mock Object Detector
    od = MagicMock()
    od.is_loaded.return_value = True
    roi = np.zeros((100, 100, 3), dtype=np.uint8)
    od.predict.return_value = (roi, 0.85, 0)
    models['od'] = od
    
    # Mock Keypoint Detector
    kd = MagicMock()
    kd.is_loaded.return_value = True
    kd.predict.return_value = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]])
    models['kd'] = kd
    
    return models


class TestRealtimePipelineInitialization:
    """Test pipeline initialization."""
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_initialize_creates_all_components(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_create_camera,
        integration_config, mock_models
    ):
        """Test that initialization creates camera, models, threads, executor."""
        # Setup mocks
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_create_camera.return_value = mock_camera
        
        mock_bc.return_value = mock_models['bc']
        mock_fs.return_value = mock_models['fs']
        mock_od.return_value = mock_models['od']
        mock_kd.return_value = mock_models['kd']
        
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize()
        
        try:
            # Verify camera initialized
            assert pipeline.camera is not None
            mock_camera.open.assert_called_once()
            
            # Verify models loaded
            assert len(pipeline.models) == 4
            
            # Verify queues created
            assert 'analysis' in pipeline.queues
            assert 'detection' in pipeline.queues
            assert 'results' in pipeline.queues
            
            # Verify threads started
            assert 'analysis' in pipeline.threads
            assert 'detection' in pipeline.threads
            
            # Verify executor created
            assert pipeline.executor is not None
            
        finally:
            pipeline._shutdown()


class TestMotionDetectionIntegration:
    """Test motion detection integration."""
    
    def test_motion_triggers_collection(self, integration_config):
        """Test that motion detection triggers frame collection."""
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize_queues()
        
        # First frame - no motion (initializes previous frame)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._check_motion(frame1)
        
        assert pipeline.collecting is False
        
        # Second frame - significant change
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pipeline._check_motion(frame2)
        
        assert pipeline.collecting is True
        assert pipeline.collect_start_frame == 0
    
    def test_cooldown_prevents_rapid_triggers(self, integration_config):
        """Test that cooldown prevents rapid motion triggers."""
        pipeline = RealtimePipeline(integration_config)
        pipeline.detection_cooldown = 5  # 5 second cooldown
        pipeline._initialize_queues()
        
        # First motion detection
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._check_motion(frame1)
        
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pipeline._check_motion(frame2)
        
        assert pipeline.collecting is True
        
        # Reset collection
        pipeline.collecting = False
        
        # Try to trigger again immediately (should be blocked by cooldown)
        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._check_motion(frame3)
        
        frame4 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pipeline._check_motion(frame4)
        
        # Should still be False due to cooldown
        assert pipeline.collecting is False


class TestFrameCollectionIntegration:
    """Test frame collection integration."""
    
    def test_collection_submits_to_queue(self, integration_config):
        """Test that frame collection submits to analysis queue."""
        pipeline = RealtimePipeline(integration_config)
        pipeline.frames_to_collect = 3
        pipeline._initialize_queues()
        pipeline.collecting = True
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Collect frames
        pipeline._collect_frame(frame)
        pipeline._collect_frame(frame)
        pipeline._collect_frame(frame)
        
        # Should have submitted to queue
        assert not pipeline.queues['analysis'].empty()
        
        # Get submitted data
        frames, start_frame = pipeline.queues['analysis'].get()
        assert len(frames) == 3
        assert start_frame == 0


class TestThreadCoordination:
    """Test thread coordination."""
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_threads_start_and_stop(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_create_camera,
        integration_config, mock_models
    ):
        """Test that threads start and stop correctly."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_create_camera.return_value = mock_camera
        
        mock_bc.return_value = mock_models['bc']
        mock_fs.return_value = mock_models['fs']
        mock_od.return_value = mock_models['od']
        mock_kd.return_value = mock_models['kd']
        
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize()
        
        # Verify threads are running
        assert pipeline.threads['analysis'].is_alive()
        assert pipeline.threads['detection'].is_alive()
        
        # Stop threads
        pipeline._stop_threads()
        
        # Give threads time to stop
        time.sleep(0.5)
        
        # Verify threads stopped
        assert not pipeline.threads.get('analysis', Mock(is_alive=lambda: True)).is_alive() or pipeline.threads == {}


class TestDetectionResultHandling:
    """Test detection result handling."""
    
    def test_handle_high_confidence_detection(self, integration_config, temp_dir):
        """Test handling of high confidence detection."""
        integration_config._config['output']['detections_dir'] = str(temp_dir / 'detections')
        
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize_queues()
        pipeline.executor = ThreadPoolExecutor(max_workers=1)
        pipeline.models['kd'] = MagicMock()
        
        # Create detection result
        result = DetectionResult(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            roi=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.85,
            class_index=0,
            frame_number=100
        )
        
        # Put result in queue
        pipeline.queues['results'].put(result)
        
        # Handle results
        with patch('crustacean.core.realtime_pipeline.save_detection') as mock_save:
            pipeline._handle_detection_results()
            
            # Verify detection was counted
            assert pipeline.detection_count == 1
            assert pipeline.latest_confidence == 0.85
        
        pipeline.executor.shutdown(wait=True)
    
    def test_handle_low_confidence_detection(self, integration_config):
        """Test handling of low confidence detection."""
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize_queues()
        pipeline.executor = ThreadPoolExecutor(max_workers=1)
        
        # Create low confidence detection result
        result = DetectionResult(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            roi=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.5,  # Below threshold
            class_index=0,
            frame_number=100
        )
        
        # Put result in queue
        pipeline.queues['results'].put(result)
        
        # Handle results
        pipeline._handle_detection_results()
        
        # Verify detection was NOT counted (low confidence)
        assert pipeline.detection_count == 0
        
        pipeline.executor.shutdown(wait=True)


class TestGracefulShutdown:
    """Test graceful shutdown."""
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_shutdown_releases_all_resources(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_create_camera,
        integration_config, mock_models
    ):
        """Test that shutdown releases all resources."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_create_camera.return_value = mock_camera
        
        mock_bc.return_value = mock_models['bc']
        mock_fs.return_value = mock_models['fs']
        mock_od.return_value = mock_models['od']
        mock_kd.return_value = mock_models['kd']
        
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize()
        
        # Verify resources exist
        assert pipeline.camera is not None
        assert pipeline.executor is not None
        assert len(pipeline.threads) > 0
        
        # Shutdown
        pipeline._shutdown()
        
        # Verify resources released
        assert pipeline.camera is None
        assert pipeline.executor is None
        assert pipeline.threads == {}
        assert pipeline.models == {}
        
        # Verify camera release was called
        mock_camera.release.assert_called_once()
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_shutdown_completes_within_timeout(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_create_camera,
        integration_config, mock_models
    ):
        """Test that shutdown completes within reasonable time."""
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        mock_create_camera.return_value = mock_camera
        
        mock_bc.return_value = mock_models['bc']
        mock_fs.return_value = mock_models['fs']
        mock_od.return_value = mock_models['od']
        mock_kd.return_value = mock_models['kd']
        
        pipeline = RealtimePipeline(integration_config)
        pipeline._initialize()
        
        # Time the shutdown
        start_time = time.time()
        pipeline._shutdown()
        shutdown_time = time.time() - start_time
        
        # Should complete within 10 seconds (requirement 15.6)
        assert shutdown_time < 10


class TestEndToEndFlow:
    """Test end-to-end flow with mocked components."""
    
    @patch('crustacean.core.realtime_pipeline.create_camera')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_full_pipeline_flow(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_create_camera,
        integration_config, mock_models
    ):
        """Test complete pipeline flow from initialization to shutdown."""
        # Setup camera mock
        mock_camera = MagicMock()
        mock_camera.open.return_value = True
        frame_count = [0]
        
        def mock_read():
            frame_count[0] += 1
            if frame_count[0] <= 10:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return None
        
        mock_camera.read.side_effect = mock_read
        mock_create_camera.return_value = mock_camera
        
        # Setup model mocks
        mock_bc.return_value = mock_models['bc']
        mock_fs.return_value = mock_models['fs']
        mock_od.return_value = mock_models['od']
        mock_kd.return_value = mock_models['kd']
        
        pipeline = RealtimePipeline(integration_config)
        
        # Initialize
        pipeline._initialize()
        
        assert pipeline.camera is not None
        assert len(pipeline.models) == 4
        assert len(pipeline.threads) == 2
        
        # Shutdown
        pipeline._shutdown()
        
        assert pipeline.camera is None
        assert pipeline.models == {}
