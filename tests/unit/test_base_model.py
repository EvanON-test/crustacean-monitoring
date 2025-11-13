"""
Unit tests for BaseModel abstract class.

Tests the base model interface including initialization, loading,
inference pipeline, context manager, and error handling.
"""

import pytest
import numpy as np
from crustacean.models.base_model import BaseModel
from crustacean.utils.config import Config
from crustacean.utils.exceptions import ModelNotLoadedError, InferenceError


# Mock model for testing
class MockModel(BaseModel):
    """Mock model implementation for testing BaseModel."""
    
    def __init__(self, config, preload=False, should_fail_load=False, should_fail_inference=False):
        self.should_fail_load = should_fail_load
        self.should_fail_inference = should_fail_inference
        self.load_called = False
        self.preprocess_called = False
        self.postprocess_called = False
        super().__init__(config, preload)
    
    def load(self):
        self.load_called = True
        if self.should_fail_load:
            raise Exception("Mock load failure")
        # Simulate TFLite interpreter
        self.interpreter = "mock_interpreter"
        self.input_details = [{'index': 0}]
        self.output_details = [{'index': 0}]
    
    def preprocess(self, input_data):
        self.preprocess_called = True
        if self.should_fail_inference:
            raise ValueError("Mock preprocess failure")
        return np.array([[input_data]])
    
    def postprocess(self, output_data):
        self.postprocess_called = True
        return output_data[0][0] * 2


class TestBaseModelInitialization:
    """Test BaseModel initialization."""
    
    def test_init_without_preload(self, sample_config_file):
        """Test initialization without preloading."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=False)
        
        assert model.config is config
        assert model.interpreter is None
        assert model.input_details is None
        assert model.output_details is None
        assert not model.is_loaded()
        assert not model.load_called
    
    def test_init_with_preload(self, sample_config_file):
        """Test initialization with preloading."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        assert model.is_loaded()
        assert model.load_called
        assert model.interpreter is not None
    
    def test_logger_created(self, sample_config_file):
        """Test that logger is created with correct name."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        assert model.logger is not None
        assert model.logger.name == 'MockModel'


class TestBaseModelLoading:
    """Test model loading and unloading."""
    
    def test_load_model(self, sample_config_file):
        """Test loading a model."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        assert not model.is_loaded()
        model.load()
        assert model.is_loaded()
        assert model.load_called
    
    def test_unload_model(self, sample_config_file):
        """Test unloading a model."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        assert model.is_loaded()
        model.unload()
        assert not model.is_loaded()
        assert model.interpreter is None
    
    def test_unload_when_not_loaded(self, sample_config_file):
        """Test unloading when model not loaded (should not error)."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        # Should not raise an error
        model.unload()
        assert not model.is_loaded()
    
    def test_is_loaded_status(self, sample_config_file):
        """Test is_loaded() returns correct status."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        assert not model.is_loaded()
        model.load()
        assert model.is_loaded()
        model.unload()
        assert not model.is_loaded()


class TestBaseModelPrediction:
    """Test model prediction pipeline."""
    
    def test_predict_raises_error_when_not_loaded(self, sample_config_file):
        """Test that predict raises error when model not loaded."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        with pytest.raises(ModelNotLoadedError) as exc_info:
            model.predict(42)
        
        assert "not loaded" in exc_info.value.message.lower()
        assert exc_info.value.details['model'] == 'MockModel'
    
    def test_predict_calls_preprocess(self, sample_config_file):
        """Test that predict calls preprocess."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        # Mock the interpreter methods
        class MockInterpreter:
            def set_tensor(self, index, data):
                pass
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.array([[21]])
        
        model.interpreter = MockInterpreter()
        
        result = model.predict(10)
        assert model.preprocess_called
    
    def test_predict_calls_postprocess(self, sample_config_file):
        """Test that predict calls postprocess."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        # Mock the interpreter methods
        class MockInterpreter:
            def set_tensor(self, index, data):
                pass
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.array([[21]])
        
        model.interpreter = MockInterpreter()
        
        result = model.predict(10)
        assert model.postprocess_called
    
    def test_predict_returns_postprocessed_result(self, sample_config_file):
        """Test that predict returns postprocessed result."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        # Mock the interpreter methods
        class MockInterpreter:
            def set_tensor(self, index, data):
                pass
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.array([[21]])
        
        model.interpreter = MockInterpreter()
        
        result = model.predict(10)
        # MockModel postprocess multiplies by 2
        assert result == 42
    
    def test_predict_raises_inference_error_on_failure(self, sample_config_file):
        """Test that predict raises InferenceError on failure."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True, should_fail_inference=True)
        
        with pytest.raises(InferenceError) as exc_info:
            model.predict(10)
        
        assert "inference failed" in exc_info.value.message.lower()
        assert exc_info.value.details['model'] == 'MockModel'


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_loads_model(self, sample_config_file):
        """Test that entering context loads model."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        assert not model.is_loaded()
        with model as m:
            assert m.is_loaded()
            assert m is model
    
    def test_context_manager_unloads_model(self, sample_config_file):
        """Test that exiting context unloads model."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        with model:
            assert model.is_loaded()
        
        assert not model.is_loaded()
    
    def test_context_manager_with_preloaded_model(self, sample_config_file):
        """Test context manager with already loaded model."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        assert model.is_loaded()
        with model:
            assert model.is_loaded()
        assert not model.is_loaded()
    
    def test_context_manager_unloads_on_exception(self, sample_config_file):
        """Test that context manager unloads even if exception occurs."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        try:
            with model:
                assert model.is_loaded()
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert not model.is_loaded()


class TestStringRepresentation:
    """Test string representation methods."""
    
    def test_repr_when_not_loaded(self, sample_config_file):
        """Test __repr__ when model not loaded."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        repr_str = repr(model)
        assert 'MockModel' in repr_str
        assert 'not loaded' in repr_str
    
    def test_repr_when_loaded(self, sample_config_file):
        """Test __repr__ when model loaded."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        repr_str = repr(model)
        assert 'MockModel' in repr_str
        assert 'loaded' in repr_str
    
    def test_str_equals_repr(self, sample_config_file):
        """Test that __str__ equals __repr__."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        assert str(model) == repr(model)


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""
    
    def test_cannot_instantiate_base_model(self, sample_config_file):
        """Test that BaseModel cannot be instantiated directly."""
        config = Config.load(str(sample_config_file))
        
        with pytest.raises(TypeError):
            BaseModel(config)
    
    def test_subclass_must_implement_load(self, sample_config_file):
        """Test that subclass must implement load()."""
        config = Config.load(str(sample_config_file))
        
        class IncompleteModel(BaseModel):
            def preprocess(self, input_data):
                pass
            def postprocess(self, output_data):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel(config)
    
    def test_subclass_must_implement_preprocess(self, sample_config_file):
        """Test that subclass must implement preprocess()."""
        config = Config.load(str(sample_config_file))
        
        class IncompleteModel(BaseModel):
            def load(self):
                pass
            def postprocess(self, output_data):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel(config)
    
    def test_subclass_must_implement_postprocess(self, sample_config_file):
        """Test that subclass must implement postprocess()."""
        config = Config.load(str(sample_config_file))
        
        class IncompleteModel(BaseModel):
            def load(self):
                pass
            def preprocess(self, input_data):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel(config)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_multiple_loads(self, sample_config_file):
        """Test calling load() multiple times."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config)
        
        model.load()
        load_count_1 = model.load_called
        model.load()
        # Should be able to call load multiple times
        assert model.is_loaded()
    
    def test_multiple_unloads(self, sample_config_file):
        """Test calling unload() multiple times."""
        config = Config.load(str(sample_config_file))
        model = MockModel(config, preload=True)
        
        model.unload()
        model.unload()  # Should not error
        assert not model.is_loaded()
