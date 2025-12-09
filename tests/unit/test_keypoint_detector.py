"""
Unit tests for KeypointDetector model.

Tests the KeypointDetector model including loading, single frame processing,
batch processing, and output shape validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from crustacean.models.keypoint_detector import KeypointDetector
from crustacean.utils.config import Config
from crustacean.utils.exceptions import ModelLoadError, InferenceError


@pytest.fixture
def keypoint_detector_config_dict():
    """
    Provide configuration for KeypointDetector testing.
    
    Returns:
        dict: Configuration with keypoint detector settings
    """
    return {
        'models': {
            'keypoint_detector': {
                'path': 'test/kd_model.tflite',
                'num_keypoints': 7
            }
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }


@pytest.fixture
def keypoint_detector_config(temp_dir, keypoint_detector_config_dict):
    """
    Create a Config object for KeypointDetector testing.
    
    Args:
        temp_dir: Temporary directory fixture
        keypoint_detector_config_dict: Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    import yaml
    config_path = temp_dir / 'kd_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(keypoint_detector_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def sample_roi():
    """
    Create a sample ROI frame for testing.
    
    Returns:
        np.ndarray: Sample grayscale ROI (539, 561)
    """
    return np.random.rand(539, 561).astype(np.float32)


@pytest.fixture
def sample_roi_batch():
    """
    Create a batch of sample ROI frames for testing.
    
    Returns:
        np.ndarray: Batch of grayscale ROIs (5, 539, 561)
    """
    return np.random.rand(5, 539, 561).astype(np.float32)


class TestKeypointDetectorInitialization:
    """Test KeypointDetector initialization."""
    
    def test_init_without_preload(self, keypoint_detector_config):
        """Test initialization without preloading."""
        kd = KeypointDetector(keypoint_detector_config, preload=False)
        
        assert kd.config is keypoint_detector_config
        assert kd.interpreter is None
        assert not kd.is_loaded()
        assert kd.num_keypoints == 7
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_init_with_preload(self, mock_interpreter, keypoint_detector_config):
        """Test initialization with preloading."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        
        assert kd.is_loaded()
        assert kd.interpreter is not None
    
    def test_config_values_loaded(self, keypoint_detector_config):
        """Test that configuration values are loaded correctly."""
        kd = KeypointDetector(keypoint_detector_config)
        
        assert kd.num_keypoints == 7
    
    def test_default_num_keypoints(self, temp_dir):
        """Test default num_keypoints when not in config."""
        import yaml
        config_dict = {
            'models': {
                'keypoint_detector': {
                    'path': 'test/model.tflite'
                }
            }
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        kd = KeypointDetector(config)
        
        assert kd.num_keypoints == 7  # Default value


class TestKeypointDetectorLoading:
    """Test model loading and unloading."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_load_model(self, mock_interpreter, keypoint_detector_config):
        """Test loading the model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        kd = KeypointDetector(keypoint_detector_config)
        kd.load()
        
        assert kd.is_loaded()
        mock_interp.allocate_tensors.assert_called_once()
    
    def test_load_missing_model_path(self, temp_dir):
        """Test that load raises error when model path missing."""
        import yaml
        config_dict = {
            'models': {
                'keypoint_detector': {}
            }
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        kd = KeypointDetector(config)
        
        with pytest.raises(ModelLoadError) as exc_info:
            kd.load()
        
        assert 'model path' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_unload_model(self, mock_interpreter, keypoint_detector_config):
        """Test unloading the model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        assert kd.is_loaded()
        
        kd.unload()
        
        assert not kd.is_loaded()
        assert kd.interpreter is None


class TestKeypointDetectorPreprocessing:
    """Test frame preprocessing."""
    
    def test_preprocess_single_frame(self, keypoint_detector_config, sample_roi):
        """Test preprocessing single frame adds batch dimension."""
        kd = KeypointDetector(keypoint_detector_config)
        
        result = kd.preprocess(sample_roi)
        
        assert result.ndim == 3
        assert result.shape[0] == 1
    
    def test_preprocess_batch(self, keypoint_detector_config, sample_roi_batch):
        """Test preprocessing batch preserves dimensions."""
        kd = KeypointDetector(keypoint_detector_config)
        
        result = kd.preprocess(sample_roi_batch)
        
        assert result.ndim == 3
        assert result.shape[0] == 5


class TestKeypointDetectorPrediction:
    """Test keypoint detection prediction."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_single_frame_shape(self, mock_interpreter, keypoint_detector_config, sample_roi):
        """Test that single frame prediction returns correct shape."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        # Mock output: 14 values (7 keypoints × 2 coords)
        mock_interp.get_tensor.return_value = np.random.rand(1, 14)
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        result = kd.predict(sample_roi)
        
        assert result.shape == (14,)
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_batch_shape(self, mock_interpreter, keypoint_detector_config, sample_roi_batch):
        """Test that batch prediction returns correct shape."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        # Mock output for each frame
        mock_interp.get_tensor.return_value = np.random.rand(1, 14)
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        result = kd.predict(sample_roi_batch)
        
        assert result.shape == (5, 14)
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_calls_interpreter(self, mock_interpreter, keypoint_detector_config, sample_roi):
        """Test that predict calls interpreter methods."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        mock_interp.get_tensor.return_value = np.random.rand(1, 14)
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        kd.predict(sample_roi)
        
        mock_interp.set_tensor.assert_called()
        mock_interp.invoke.assert_called()
        mock_interp.get_tensor.assert_called()
    
    def test_predict_raises_error_when_not_loaded(self, keypoint_detector_config, sample_roi):
        """Test that predict raises error when model not loaded."""
        kd = KeypointDetector(keypoint_detector_config)
        
        with pytest.raises(InferenceError) as exc_info:
            kd.predict(sample_roi)
        
        assert 'not loaded' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_output_values_range(self, mock_interpreter, keypoint_detector_config, sample_roi):
        """Test that output values are reasonable coordinates."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        
        # Mock realistic coordinate output
        mock_output = np.array([[100, 150, 200, 180, 250, 200, 300, 220, 
                                 350, 240, 400, 260, 450, 280]])
        mock_interp.get_tensor.return_value = mock_output
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        result = kd.predict(sample_roi)
        
        assert len(result) == 14
        assert np.array_equal(result, mock_output.flatten())


class TestKeypointPairs:
    """Test keypoint pairs utility method."""
    
    def test_get_keypoint_pairs_single(self, keypoint_detector_config):
        """Test reshaping single frame coordinates to pairs."""
        kd = KeypointDetector(keypoint_detector_config)
        
        coords = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        pairs = kd.get_keypoint_pairs(coords)
        
        assert pairs.shape == (7, 2)
        assert np.array_equal(pairs[0], [10, 20])
        assert np.array_equal(pairs[6], [130, 140])
    
    def test_get_keypoint_pairs_batch(self, keypoint_detector_config):
        """Test reshaping batch coordinates to pairs."""
        kd = KeypointDetector(keypoint_detector_config)
        
        coords = np.random.rand(5, 14)
        pairs = kd.get_keypoint_pairs(coords)
        
        assert pairs.shape == (5, 7, 2)


class TestContextManager:
    """Test context manager functionality."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_context_manager_loads_and_unloads(self, mock_interpreter, keypoint_detector_config):
        """Test that context manager loads and unloads model."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        
        kd = KeypointDetector(keypoint_detector_config)
        
        assert not kd.is_loaded()
        
        with kd:
            assert kd.is_loaded()
        
        assert not kd.is_loaded()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_single_frame_batch_dimension(self, mock_interpreter, keypoint_detector_config):
        """Test single frame with explicit batch dimension."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        mock_interp.get_tensor.return_value = np.random.rand(1, 14)
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        
        # Single frame with batch dimension (1, h, w)
        single_batch = np.random.rand(1, 539, 561).astype(np.float32)
        result = kd.predict(single_batch)
        
        # Should still return single frame result
        assert result.shape == (14,)
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_different_roi_sizes(self, mock_interpreter, keypoint_detector_config):
        """Test with different ROI sizes."""
        mock_interp = MagicMock()
        mock_interpreter.return_value = mock_interp
        mock_interp.get_input_details.return_value = [{'index': 0}]
        mock_interp.get_output_details.return_value = [{'index': 0}]
        mock_interp.get_tensor.return_value = np.random.rand(1, 14)
        
        kd = KeypointDetector(keypoint_detector_config, preload=True)
        
        # Different size ROI
        small_roi = np.random.rand(100, 100).astype(np.float32)
        result = kd.predict(small_roi)
        
        assert result.shape == (14,)
    
    def test_empty_batch(self, keypoint_detector_config):
        """Test preprocessing empty batch."""
        kd = KeypointDetector(keypoint_detector_config)
        
        empty_batch = np.zeros((0, 539, 561))
        result = kd.preprocess(empty_batch)
        
        assert result.shape[0] == 0
