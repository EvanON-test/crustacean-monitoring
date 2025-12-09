"""
Unit tests for ObjectDetector model.

Tests the ObjectDetector model including loading, preprocessing (padding, resizing),
NMS application, ROI cropping, and confidence thresholding.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from crustacean.models.object_detector import (
    ObjectDetector, 
    non_max_suppression, 
    xywh2xyxy, 
    calculate_iou
)
from crustacean.utils.config import Config
from crustacean.utils.exceptions import ModelLoadError, InferenceError


@pytest.fixture
def object_detector_config_dict():
    """
    Provide configuration for ObjectDetector testing.
    
    Returns:
        dict: Configuration with object detector settings
    """
    return {
        'models': {
            'object_detector': {
                'path': 'test/od_model.tflite',
                'input_size': 640,
                'confidence_threshold': 0.75,
                'fixed_crop_width': 539,
                'fixed_crop_height': 561
            }
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }


@pytest.fixture
def object_detector_config(temp_dir, object_detector_config_dict):
    """
    Create a Config object for ObjectDetector testing.
    
    Args:
        temp_dir: Temporary directory fixture
        object_detector_config_dict: Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    import yaml
    config_path = temp_dir / 'od_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(object_detector_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def sample_frame():
    """
    Create a sample BGR frame for testing.
    
    Returns:
        np.ndarray: Sample frame (720, 1280, 3)
    """
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


class TestHelperFunctions:
    """Test helper functions for object detection."""
    
    def test_xywh2xyxy_single_box(self):
        """Test xywh to xyxy conversion for single box."""
        # Box at center (100, 100) with width 50, height 40
        xywh = np.array([[100, 100, 50, 40]])
        xyxy = xywh2xyxy(xywh)
        
        # Expected: x1=75, y1=80, x2=125, y2=120
        assert xyxy[0, 0] == 75   # x1 = 100 - 50/2
        assert xyxy[0, 1] == 80   # y1 = 100 - 40/2
        assert xyxy[0, 2] == 125  # x2 = 100 + 50/2
        assert xyxy[0, 3] == 120  # y2 = 100 + 40/2
    
    def test_xywh2xyxy_multiple_boxes(self):
        """Test xywh to xyxy conversion for multiple boxes."""
        xywh = np.array([
            [100, 100, 50, 40],
            [200, 200, 100, 80]
        ])
        xyxy = xywh2xyxy(xywh)
        
        assert xyxy.shape == (2, 4)
        assert xyxy[1, 0] == 150  # x1 = 200 - 100/2
        assert xyxy[1, 1] == 160  # y1 = 200 - 80/2
    
    def test_calculate_iou_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        box = np.array([0, 0, 100, 100])
        boxes = np.array([[0, 0, 100, 100]])
        
        iou = calculate_iou(box, boxes)
        
        assert np.isclose(iou[0], 1.0)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        box = np.array([0, 0, 50, 50])
        boxes = np.array([[100, 100, 150, 150]])
        
        iou = calculate_iou(box, boxes)
        
        assert iou[0] == 0.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        box = np.array([0, 0, 100, 100])
        boxes = np.array([[50, 50, 150, 150]])
        
        iou = calculate_iou(box, boxes)
        
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.143
        assert 0.1 < iou[0] < 0.2


class TestNonMaxSuppression:
    """Test Non-Maximum Suppression function."""
    
    def test_nms_single_detection(self):
        """Test NMS with single detection."""
        # Create prediction with one detection
        # Format: [x, y, w, h, objectness, class0_conf, class1_conf]
        prediction = np.array([[[
            320, 320, 100, 100, 0.9, 0.8, 0.1
        ]]])
        
        result = non_max_suppression(prediction, conf_thres=0.25)
        
        assert len(result) == 1
        assert len(result[0]) == 1
    
    def test_nms_filters_low_confidence(self):
        """Test that NMS filters low confidence detections."""
        prediction = np.array([[[
            320, 320, 100, 100, 0.1, 0.05, 0.05  # Low confidence
        ]]])
        
        result = non_max_suppression(prediction, conf_thres=0.25)
        
        assert len(result[0]) == 0
    
    def test_nms_returns_max_det(self):
        """Test that NMS respects max_det parameter."""
        # Create multiple high-confidence detections
        prediction = np.array([[
            [100, 100, 50, 50, 0.9, 0.8, 0.1],
            [200, 200, 50, 50, 0.85, 0.75, 0.1],
            [300, 300, 50, 50, 0.8, 0.7, 0.1]
        ]])
        
        result = non_max_suppression(prediction, conf_thres=0.25, max_det=1)
        
        assert len(result[0]) <= 1


class TestObjectDetectorInitialization:
    """Test ObjectDetector initialization."""
    
    def test_init_without_preload(self, object_detector_config):
        """Test initialization without preloading."""
        od = ObjectDetector(object_detector_config, preload=False)
        
        assert od.config is object_detector_config
        assert od.interpreter is None
        assert not od.is_loaded()
        assert od.input_size == 640
        assert od.confidence_threshold == 0.75
        assert od.fixed_crop_width == 539
        assert od.fixed_crop_height == 561
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_init_with_preload(self, mock_interpreter, object_detector_config):
        """Test initialization with preloading."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        od = ObjectDetector(object_detector_config, preload=True)
        
        assert od.is_loaded()
        assert od.interpreter is not None
    
    def test_config_values_loaded(self, object_detector_config):
        """Test that configuration values are loaded correctly."""
        od = ObjectDetector(object_detector_config)
        
        assert od.input_size == 640
        assert od.confidence_threshold == 0.75
        assert od.fixed_crop_width == 539
        assert od.fixed_crop_height == 561


class TestObjectDetectorLoading:
    """Test model loading and unloading."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_load_model(self, mock_interpreter, object_detector_config):
        """Test loading the model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        od = ObjectDetector(object_detector_config)
        od.load()
        
        assert od.is_loaded()
        mock_interp.allocate_tensors.assert_called_once()
    
    def test_load_missing_model_path(self, temp_dir):
        """Test that load raises error when model path missing."""
        import yaml
        config_dict = {
            'models': {
                'object_detector': {}
            }
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        od = ObjectDetector(config)
        
        with pytest.raises(ModelLoadError) as exc_info:
            od.load()
        
        assert 'model path' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_unload_model(self, mock_interpreter, object_detector_config):
        """Test unloading the model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        od = ObjectDetector(object_detector_config, preload=True)
        assert od.is_loaded()
        
        od.unload()
        
        assert not od.is_loaded()
        assert od.interpreter is None


class TestObjectDetectorPreprocessing:
    """Test frame preprocessing."""
    
    def test_preprocess_returns_tuple(self, object_detector_config, sample_frame):
        """Test that preprocess returns tuple of input and original frame."""
        od = ObjectDetector(object_detector_config)
        
        result = od.preprocess(sample_frame)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_preprocess_output_shape(self, object_detector_config, sample_frame):
        """Test that preprocessed output has correct shape."""
        od = ObjectDetector(object_detector_config)
        
        input_data, _ = od.preprocess(sample_frame)
        
        # Should be (1, 640, 640, 3)
        assert input_data.shape == (1, 640, 640, 3)
    
    def test_preprocess_preserves_original(self, object_detector_config, sample_frame):
        """Test that preprocess preserves original frame."""
        od = ObjectDetector(object_detector_config)
        original_copy = sample_frame.copy()
        
        _, preserved = od.preprocess(sample_frame)
        
        assert np.array_equal(preserved, original_copy)


class TestObjectDetectorROICropping:
    """Test ROI cropping functionality."""
    
    def test_crop_roi_shape(self, object_detector_config):
        """Test that cropped ROI has correct shape."""
        od = ObjectDetector(object_detector_config)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        roi = od._crop_roi(frame, 100, 100)
        
        assert roi.shape == (539, 561)
    
    def test_crop_roi_grayscale(self, object_detector_config):
        """Test that cropped ROI is grayscale."""
        od = ObjectDetector(object_detector_config)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        roi = od._crop_roi(frame, 100, 100)
        
        # Should be 2D (grayscale)
        assert len(roi.shape) == 2
    
    def test_crop_roi_handles_boundary(self, object_detector_config):
        """Test that cropping handles frame boundaries."""
        od = ObjectDetector(object_detector_config)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Crop near edge - should not raise error
        roi = od._crop_roi(frame, 600, 1000)
        
        assert roi.shape == (539, 561)


class TestObjectDetectorPrediction:
    """Test object detection prediction."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_returns_tuple(self, mock_interpreter, object_detector_config, sample_frame):
        """Test that predict returns tuple of (roi, confidence, class_index)."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        # Mock detection output
        # Format: [x, y, w, h, objectness, class0_conf, class1_conf]
        mock_output = np.array([[[
            320, 320, 100, 100, 0.9, 0.8, 0.1
        ]]])
        mock_interp.get_tensor.return_value = mock_output
        
        od = ObjectDetector(object_detector_config, preload=True)
        result = od.predict(sample_frame)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_roi_shape(self, mock_interpreter, object_detector_config, sample_frame):
        """Test that predicted ROI has correct shape."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        mock_output = np.array([[[
            320, 320, 100, 100, 0.9, 0.8, 0.1
        ]]])
        mock_interp.get_tensor.return_value = mock_output
        
        od = ObjectDetector(object_detector_config, preload=True)
        roi, _, _ = od.predict(sample_frame)
        
        assert roi.shape == (539, 561)
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_confidence_range(self, mock_interpreter, object_detector_config, sample_frame):
        """Test that confidence is in valid range."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        mock_output = np.array([[[
            320, 320, 100, 100, 0.9, 0.8, 0.1
        ]]])
        mock_interp.get_tensor.return_value = mock_output
        
        od = ObjectDetector(object_detector_config, preload=True)
        _, confidence, _ = od.predict(sample_frame)
        
        assert 0.0 <= confidence <= 1.0
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_class_index(self, mock_interpreter, object_detector_config, sample_frame):
        """Test that class index is valid."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        mock_output = np.array([[[
            320, 320, 100, 100, 0.9, 0.8, 0.1
        ]]])
        mock_interp.get_tensor.return_value = mock_output
        
        od = ObjectDetector(object_detector_config, preload=True)
        _, _, class_index = od.predict(sample_frame)
        
        assert class_index in [0, 1, -1]  # 0=crab, 1=lobster, -1=no detection
    
    def test_predict_raises_error_when_not_loaded(self, object_detector_config, sample_frame):
        """Test that predict raises error when model not loaded."""
        od = ObjectDetector(object_detector_config)
        
        with pytest.raises(InferenceError) as exc_info:
            od.predict(sample_frame)
        
        assert 'not loaded' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_no_detection(self, mock_interpreter, object_detector_config, sample_frame):
        """Test prediction when no detection found."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        # Low confidence detection that will be filtered
        mock_output = np.array([[[
            320, 320, 100, 100, 0.1, 0.05, 0.05
        ]]])
        mock_interp.get_tensor.return_value = mock_output
        
        od = ObjectDetector(object_detector_config, preload=True)
        roi, confidence, class_index = od.predict(sample_frame)
        
        # Should return empty ROI with zero confidence
        assert confidence == 0.0
        assert class_index == -1
        assert roi.shape == (539, 561)


class TestContextManager:
    """Test context manager functionality."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_context_manager_loads_and_unloads(self, mock_interpreter, object_detector_config):
        """Test that context manager loads and unloads model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        od = ObjectDetector(object_detector_config)
        
        assert not od.is_loaded()
        
        with od:
            assert od.is_loaded()
        
        assert not od.is_loaded()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_frame(self, object_detector_config):
        """Test preprocessing with small frame."""
        od = ObjectDetector(object_detector_config)
        small_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        input_data, _ = od.preprocess(small_frame)
        
        # Should still produce correct output shape
        assert input_data.shape == (1, 640, 640, 3)
    
    def test_large_frame(self, object_detector_config):
        """Test preprocessing with large frame."""
        od = ObjectDetector(object_detector_config)
        large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        input_data, _ = od.preprocess(large_frame)
        
        # Should still produce correct output shape
        assert input_data.shape == (1, 640, 640, 3)
