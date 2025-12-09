"""
Unit tests for base Pipeline class.

Tests the Pipeline abstract class including initialization, model loading,
cleanup, and context manager functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from crustacean.core.pipeline import Pipeline
from crustacean.utils.config import Config


# Concrete implementation for testing
class MockPipeline(Pipeline):
    """Mock pipeline implementation for testing."""
    
    def __init__(self, config, should_fail_run=False):
        super().__init__(config)
        self.should_fail_run = should_fail_run
        self.run_called = False
    
    def run(self):
        self.run_called = True
        if self.should_fail_run:
            raise RuntimeError("Mock run failure")


@pytest.fixture
def pipeline_config_dict():
    """
    Provide configuration for Pipeline testing.
    
    Returns:
        dict: Configuration with pipeline settings
    """
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
                'input_size': 640
            },
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
def pipeline_config(temp_dir, pipeline_config_dict):
    """
    Create a Config object for Pipeline testing.
    
    Args:
        temp_dir: Temporary directory fixture
        pipeline_config_dict: Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    import yaml
    config_path = temp_dir / 'pipeline_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(pipeline_config_dict, f)
    return Config.load(str(config_path))


class TestPipelineInitialization:
    """Test Pipeline initialization."""
    
    def test_init_sets_config(self, pipeline_config):
        """Test that initialization sets config."""
        pipeline = MockPipeline(pipeline_config)
        
        assert pipeline.config is pipeline_config
    
    def test_init_creates_logger(self, pipeline_config):
        """Test that initialization creates logger."""
        pipeline = MockPipeline(pipeline_config)
        
        assert pipeline.logger is not None
        assert pipeline.logger.name == 'MockPipeline'
    
    def test_init_empty_models(self, pipeline_config):
        """Test that initialization starts with empty models dict."""
        pipeline = MockPipeline(pipeline_config)
        
        assert pipeline.models == {}


class TestPipelineModelLoading:
    """Test model loading functionality."""
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_load_models_creates_all_models(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that load_models creates all four models."""
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=False)
        
        assert 'bc' in pipeline.models
        assert 'fs' in pipeline.models
        assert 'od' in pipeline.models
        assert 'kd' in pipeline.models
        assert len(pipeline.models) == 4
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_load_models_passes_preload_flag(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that load_models passes preload flag to models."""
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        mock_bc.assert_called_once_with(pipeline_config, preload=True)
        mock_fs.assert_called_once_with(pipeline_config, preload=True)
        mock_od.assert_called_once_with(pipeline_config, preload=True)
        mock_kd.assert_called_once_with(pipeline_config, preload=True)
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_load_models_without_preload(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that load_models can be called without preload."""
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=False)
        
        mock_bc.assert_called_once_with(pipeline_config, preload=False)


class TestPipelineCleanup:
    """Test cleanup functionality."""
    
    def test_cleanup_empty_models(self, pipeline_config):
        """Test cleanup with no models loaded."""
        pipeline = MockPipeline(pipeline_config)
        
        # Should not raise error
        pipeline.cleanup()
        
        assert pipeline.models == {}
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_cleanup_unloads_models(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that cleanup unloads all models."""
        # Setup mock models
        mock_bc_instance = MagicMock()
        mock_bc_instance.is_loaded.return_value = True
        mock_bc.return_value = mock_bc_instance
        
        mock_fs_instance = MagicMock()
        mock_fs_instance.is_loaded.return_value = True
        mock_fs.return_value = mock_fs_instance
        
        mock_od_instance = MagicMock()
        mock_od_instance.is_loaded.return_value = True
        mock_od.return_value = mock_od_instance
        
        mock_kd_instance = MagicMock()
        mock_kd_instance.is_loaded.return_value = True
        mock_kd.return_value = mock_kd_instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        pipeline.cleanup()
        
        mock_bc_instance.unload.assert_called_once()
        mock_fs_instance.unload.assert_called_once()
        mock_od_instance.unload.assert_called_once()
        mock_kd_instance.unload.assert_called_once()
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_cleanup_clears_models_dict(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that cleanup clears models dictionary."""
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=False)
        
        assert len(pipeline.models) == 4
        
        pipeline.cleanup()
        
        assert pipeline.models == {}
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_cleanup_handles_unload_error(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that cleanup handles errors during unload."""
        mock_bc_instance = MagicMock()
        mock_bc_instance.is_loaded.return_value = True
        mock_bc_instance.unload.side_effect = Exception("Unload error")
        mock_bc.return_value = mock_bc_instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        # Should not raise error
        pipeline.cleanup()
        
        assert pipeline.models == {}


class TestPipelineGetModel:
    """Test get_model functionality."""
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_get_model_returns_model(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that get_model returns correct model."""
        mock_bc_instance = MagicMock()
        mock_bc.return_value = mock_bc_instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=False)
        
        result = pipeline.get_model('bc')
        
        assert result is mock_bc_instance
    
    def test_get_model_returns_none_for_unknown(self, pipeline_config):
        """Test that get_model returns None for unknown model."""
        pipeline = MockPipeline(pipeline_config)
        
        result = pipeline.get_model('unknown')
        
        assert result is None


class TestPipelineIsModelsLoaded:
    """Test is_models_loaded functionality."""
    
    def test_is_models_loaded_empty(self, pipeline_config):
        """Test is_models_loaded with no models."""
        pipeline = MockPipeline(pipeline_config)
        
        assert not pipeline.is_models_loaded()
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_is_models_loaded_all_loaded(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test is_models_loaded when all models are loaded."""
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        assert pipeline.is_models_loaded()
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_is_models_loaded_partial(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test is_models_loaded when some models not loaded."""
        mock_bc_instance = MagicMock()
        mock_bc_instance.is_loaded.return_value = True
        mock_bc.return_value = mock_bc_instance
        
        mock_fs_instance = MagicMock()
        mock_fs_instance.is_loaded.return_value = False  # Not loaded
        mock_fs.return_value = mock_fs_instance
        
        for mock_class in [mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        assert not pipeline.is_models_loaded()


class TestPipelineContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_returns_self(self, pipeline_config):
        """Test that context manager returns pipeline."""
        pipeline = MockPipeline(pipeline_config)
        
        with pipeline as p:
            assert p is pipeline
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_context_manager_calls_cleanup(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that context manager calls cleanup on exit."""
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        assert len(pipeline.models) == 4
        
        with pipeline:
            pass
        
        assert pipeline.models == {}
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_context_manager_cleanup_on_exception(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test that context manager cleans up even on exception."""
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = MockPipeline(pipeline_config, should_fail_run=True)
        pipeline.load_models(preload=True)
        
        try:
            with pipeline:
                pipeline.run()
        except RuntimeError:
            pass
        
        assert pipeline.models == {}


class TestPipelineAbstract:
    """Test abstract method requirements."""
    
    def test_cannot_instantiate_pipeline_directly(self, pipeline_config):
        """Test that Pipeline cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Pipeline(pipeline_config)
    
    def test_subclass_must_implement_run(self, pipeline_config):
        """Test that subclass must implement run()."""
        class IncompletePipeline(Pipeline):
            pass
        
        with pytest.raises(TypeError):
            IncompletePipeline(pipeline_config)


class TestPipelineRepr:
    """Test string representation."""
    
    def test_repr_not_loaded(self, pipeline_config):
        """Test repr when models not loaded."""
        pipeline = MockPipeline(pipeline_config)
        
        repr_str = repr(pipeline)
        
        assert 'MockPipeline' in repr_str
        assert 'not loaded' in repr_str
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_repr_loaded(
        self, mock_kd, mock_od, mock_fs, mock_bc, pipeline_config
    ):
        """Test repr when models loaded."""
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = MockPipeline(pipeline_config)
        pipeline.load_models(preload=True)
        
        repr_str = repr(pipeline)
        
        assert 'MockPipeline' in repr_str
        assert 'loaded' in repr_str
