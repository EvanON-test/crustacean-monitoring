"""
Unit tests for FrameSelector model.

Tests the FrameSelector model including loading both models,
frame quality prediction, contig detection logic, and best frame selection.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch
from crustacean.models.frame_selector import FrameSelector
from crustacean.utils.config import Config
from crustacean.utils.exceptions import ModelLoadError, InferenceError


@pytest.fixture
def frame_selector_config_dict():
    """
    Provide configuration for FrameSelector testing.
    
    Returns:
        dict: Configuration with frame selector settings
    """
    return {
        'models': {
            'frame_selector': {
                'top_model_path': 'test/top_model.tflite',
                'bottom_model_path': 'test/bottom_model.tflite',
                'input_width': 320,
                'input_height': 180
            }
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }


@pytest.fixture
def frame_selector_config(temp_dir, frame_selector_config_dict):
    """
    Create a Config object for FrameSelector testing.
    
    Args:
        temp_dir: Temporary directory fixture
        frame_selector_config_dict: Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    import yaml
    config_path = temp_dir / 'fs_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(frame_selector_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def mock_video_capture():
    """
    Create a mock VideoCapture object.
    
    Returns:
        Mock: Mock VideoCapture with read() method
    """
    mock_video = Mock(spec=cv2.VideoCapture)
    
    # Create test frames (BGR format)
    test_frames = []
    for i in range(10):
        # Create different frames with varying brightness
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * (i * 25)
        test_frames.append(frame)
    
    # Setup read() to return frames sequentially
    mock_video.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
    mock_video.get.return_value = len(test_frames)
    
    return mock_video


class TestFrameSelectorInitialization:
    """Test FrameSelector initialization."""
    
    def test_init_without_preload(self, frame_selector_config):
        """Test initialization without preloading."""
        fs = FrameSelector(frame_selector_config, preload=False)
        
        assert fs.config is frame_selector_config
        assert fs.top_interpreter is None
        assert fs.bottom_interpreter is None
        assert not fs.is_loaded()
        assert fs.input_width == 320
        assert fs.input_height == 180
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_init_with_preload(self, mock_interpreter, frame_selector_config):
        """Test initialization with preloading."""
        # Setup mock interpreters
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        assert fs.is_loaded()
        assert fs.top_interpreter is not None
        assert fs.bottom_interpreter is not None
    
    def test_config_values_loaded(self, frame_selector_config):
        """Test that configuration values are loaded correctly."""
        fs = FrameSelector(frame_selector_config)
        
        assert fs.input_width == 320
        assert fs.input_height == 180


class TestFrameSelectorLoading:
    """Test model loading and unloading."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_load_both_models(self, mock_interpreter, frame_selector_config):
        """Test loading both top and bottom models."""
        # Setup mock interpreters
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Setup mock details
        mock_top.get_input_details.return_value = [{'index': 0}]
        mock_top.get_output_details.return_value = [{'index': 0}]
        mock_bottom.get_input_details.return_value = [{'index': 0}]
        mock_bottom.get_output_details.return_value = [{'index': 0}]
        
        fs = FrameSelector(frame_selector_config)
        fs.load()
        
        assert fs.is_loaded()
        assert fs.top_interpreter is mock_top
        assert fs.bottom_interpreter is mock_bottom
        assert mock_interpreter.call_count == 2
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_load_calls_allocate_tensors(self, mock_interpreter, frame_selector_config):
        """Test that load() calls allocate_tensors on both models."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config)
        fs.load()
        
        mock_top.allocate_tensors.assert_called_once()
        mock_bottom.allocate_tensors.assert_called_once()
    
    def test_load_missing_top_model_path(self, temp_dir):
        """Test that load raises error when top model path missing."""
        import yaml
        config_dict = {
            'models': {
                'frame_selector': {
                    'bottom_model_path': 'test/bottom.tflite'
                }
            }
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        fs = FrameSelector(config)
        
        with pytest.raises(ModelLoadError) as exc_info:
            fs.load()
        
        assert 'top model path' in exc_info.value.message.lower()
    
    def test_load_missing_bottom_model_path(self, temp_dir):
        """Test that load raises error when bottom model path missing."""
        import yaml
        config_dict = {
            'models': {
                'frame_selector': {
                    'top_model_path': 'test/top.tflite'
                }
            }
        }
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        config = Config.load(str(config_path))
        
        fs = FrameSelector(config)
        
        with pytest.raises(ModelLoadError) as exc_info:
            fs.load()
        
        assert 'bottom model path' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_unload_models(self, mock_interpreter, frame_selector_config):
        """Test unloading both models."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config, preload=True)
        assert fs.is_loaded()
        
        fs.unload()
        
        assert not fs.is_loaded()
        assert fs.top_interpreter is None
        assert fs.bottom_interpreter is None
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_is_loaded_checks_both_models(self, mock_interpreter, frame_selector_config):
        """Test that is_loaded() checks both models."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config)
        assert not fs.is_loaded()
        
        fs.load()
        assert fs.is_loaded()
        
        # Manually set one to None
        fs.top_interpreter = None
        assert not fs.is_loaded()


class TestFrameSelectorPreprocessing:
    """Test frame preprocessing."""
    
    def test_preprocess_converts_to_grayscale(self, frame_selector_config):
        """Test that preprocess converts BGR to grayscale."""
        fs = FrameSelector(frame_selector_config)
        
        # Create a BGR frame
        bgr_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        result = fs.preprocess(bgr_frame)
        
        # Result should have shape (1, height, width, 1)
        assert result.shape == (1, 180, 320, 1)
    
    def test_preprocess_rescales_frame(self, frame_selector_config):
        """Test that preprocess rescales to configured dimensions."""
        fs = FrameSelector(frame_selector_config)
        
        # Create a frame with different dimensions
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        result = fs.preprocess(frame)
        
        # Should be rescaled to 320x180
        assert result.shape[1] == 180  # height
        assert result.shape[2] == 320  # width
    
    def test_preprocess_reshapes_for_tflite(self, frame_selector_config):
        """Test that preprocess reshapes for TFLite format."""
        fs = FrameSelector(frame_selector_config)
        
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        result = fs.preprocess(frame)
        
        # Should have batch dimension and channel dimension
        assert result.shape[0] == 1  # batch
        assert result.shape[3] == 1  # channels (grayscale)


class TestFrameSelectorPostprocessing:
    """Test output postprocessing."""
    
    def test_postprocess_returns_float(self, frame_selector_config):
        """Test that postprocess returns a float quality score."""
        fs = FrameSelector(frame_selector_config)
        
        # Simulate model output
        output = np.array([[0.85]])
        
        result = fs.postprocess(output)
        
        assert isinstance(result, float)
        assert result == 0.85


class TestFrameSelectorPrediction:
    """Test frame selection prediction."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_single_segment(self, mock_interpreter, frame_selector_config, mock_video_capture):
        """Test prediction with a single continuous segment."""
        # Setup mock interpreters
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Setup quality predictions (frame 2 has highest quality)
        mock_top.get_tensor.side_effect = [
            np.array([[0.5]]),  # frame 1
            np.array([[0.9]]),  # frame 2 (best)
            np.array([[0.6]])   # frame 3
        ]
        mock_bottom.get_tensor.side_effect = [
            np.array([[0.4]]),  # frame 1
            np.array([[0.7]]),  # frame 2
            np.array([[0.8]])   # frame 3 (best)
        ]
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: frames 1, 2, 3 have crustacean
        signal = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        
        result = fs.predict(signal, mock_video_capture)
        
        # Should return [[2], [3]] - best frames for top and bottom
        assert len(result) == 2
        assert len(result[0]) == 1  # One segment
        assert len(result[1]) == 1  # One segment
        assert result[0][0] == 2  # Top model best frame
        assert result[1][0] == 3  # Bottom model best frame
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_multiple_segments(self, mock_interpreter, frame_selector_config):
        """Test prediction with multiple continuous segments."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Setup quality predictions for two segments
        mock_top.get_tensor.side_effect = [
            np.array([[0.5]]),  # segment 1, frame 1
            np.array([[0.9]]),  # segment 1, frame 2 (best)
            np.array([[0.6]]),  # segment 2, frame 5
            np.array([[0.8]])   # segment 2, frame 6 (best)
        ]
        mock_bottom.get_tensor.side_effect = [
            np.array([[0.7]]),  # segment 1, frame 1 (best)
            np.array([[0.5]]),  # segment 1, frame 2
            np.array([[0.9]]),  # segment 2, frame 5 (best)
            np.array([[0.6]])   # segment 2, frame 6
        ]
        
        # Create mock video with frames
        mock_video = Mock(spec=cv2.VideoCapture)
        test_frames = [np.ones((720, 1280, 3), dtype=np.uint8) * i for i in range(10)]
        mock_video.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_video.get.return_value = 10
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: two segments (1-2 and 5-6)
        signal = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        
        result = fs.predict(signal, mock_video)
        
        # Should return two segments for each model
        assert len(result) == 2
        assert len(result[0]) == 2  # Two segments
        assert len(result[1]) == 2  # Two segments
        assert result[0] == [2, 6]  # Top model best frames
        assert result[1] == [1, 5]  # Bottom model best frames
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_segment_at_end(self, mock_interpreter, frame_selector_config):
        """Test prediction when segment extends to end of video."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Setup quality predictions
        mock_top.get_tensor.side_effect = [
            np.array([[0.5]]),
            np.array([[0.9]]),  # best
            np.array([[0.6]])
        ]
        mock_bottom.get_tensor.side_effect = [
            np.array([[0.7]]),  # best
            np.array([[0.5]]),
            np.array([[0.6]])
        ]
        
        # Create mock video
        mock_video = Mock(spec=cv2.VideoCapture)
        test_frames = [np.ones((720, 1280, 3), dtype=np.uint8) * i for i in range(10)]
        mock_video.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_video.get.return_value = 10
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: segment at end (frames 7-9)
        signal = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        
        result = fs.predict(signal, mock_video)
        
        # Should handle segment at end
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0] == 8  # Top model best
        assert result[1][0] == 7  # Bottom model best
    
    def test_predict_raises_error_when_not_loaded(self, frame_selector_config, mock_video_capture):
        """Test that predict raises error when models not loaded."""
        fs = FrameSelector(frame_selector_config)
        signal = np.array([0, 1, 1, 0])
        
        with pytest.raises(InferenceError) as exc_info:
            fs.predict(signal, mock_video_capture)
        
        assert 'not loaded' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_handles_video_read_failure(self, mock_interpreter, frame_selector_config):
        """Test that predict handles video read failure."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Create mock video that fails to read
        mock_video = Mock(spec=cv2.VideoCapture)
        mock_video.read.return_value = (False, None)
        
        fs = FrameSelector(frame_selector_config, preload=True)
        signal = np.array([0, 1, 1, 0])
        
        with pytest.raises(InferenceError) as exc_info:
            fs.predict(signal, mock_video)
        
        assert 'failed to read' in exc_info.value.message.lower()
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_predict_no_segments(self, mock_interpreter, frame_selector_config, mock_video_capture):
        """Test prediction when no crustaceans detected (all zeros)."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: no crustaceans
        signal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        result = fs.predict(signal, mock_video_capture)
        
        # Should return empty lists
        assert result == [[], []]


class TestContextManager:
    """Test context manager functionality."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_context_manager_loads_and_unloads(self, mock_interpreter, frame_selector_config):
        """Test that context manager loads and unloads models."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        fs = FrameSelector(frame_selector_config)
        
        assert not fs.is_loaded()
        
        with fs:
            assert fs.is_loaded()
        
        assert not fs.is_loaded()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_single_frame_segment(self, mock_interpreter, frame_selector_config):
        """Test segment with only one frame."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        mock_top.get_tensor.return_value = np.array([[0.8]])
        mock_bottom.get_tensor.return_value = np.array([[0.7]])
        
        # Create mock video
        mock_video = Mock(spec=cv2.VideoCapture)
        test_frames = [np.ones((720, 1280, 3), dtype=np.uint8) * i for i in range(5)]
        mock_video.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_video.get.return_value = 5
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: single frame segment
        signal = np.array([0, 0, 1, 0, 0])
        
        result = fs.predict(signal, mock_video)
        
        # Should select that single frame
        assert result[0] == [2]
        assert result[1] == [2]
    
    @patch('tflite_runtime.interpreter.Interpreter')
    def test_all_frames_have_crustacean(self, mock_interpreter, frame_selector_config):
        """Test when all frames have crustacean (one big segment)."""
        mock_top = MagicMock()
        mock_bottom = MagicMock()
        mock_interpreter.side_effect = [mock_top, mock_bottom]
        
        # Frame 2 has highest quality
        mock_top.get_tensor.side_effect = [
            np.array([[0.5]]),
            np.array([[0.6]]),
            np.array([[0.9]]),  # best
            np.array([[0.4]]),
            np.array([[0.7]])
        ]
        mock_bottom.get_tensor.side_effect = [
            np.array([[0.5]]),
            np.array([[0.8]]),  # best
            np.array([[0.6]]),
            np.array([[0.4]]),
            np.array([[0.7]])
        ]
        
        # Create mock video
        mock_video = Mock(spec=cv2.VideoCapture)
        test_frames = [np.ones((720, 1280, 3), dtype=np.uint8) * i for i in range(5)]
        mock_video.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_video.get.return_value = 5
        
        fs = FrameSelector(frame_selector_config, preload=True)
        
        # Signal: all frames have crustacean
        signal = np.array([1, 1, 1, 1, 1])
        
        result = fs.predict(signal, mock_video)
        
        # Should return one segment with best frames
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0] == 2  # Top best
        assert result[1][0] == 1  # Bottom best
