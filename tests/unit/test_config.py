"""
Unit tests for configuration management.

Tests the Config class including loading, dot notation access,
environment variable overrides, and error handling.
"""

import pytest
import os
import yaml
from pathlib import Path
from crustacean.utils.config import Config


class TestConfigLoading:
    """Test configuration file loading."""
    
    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid configuration file."""
        config = Config.load(str(sample_config_file))
        assert config is not None
        assert isinstance(config, Config)
    
    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = Config.load('config/default_config.yaml')
        assert config is not None
        # Verify key sections exist
        assert config.get('models') is not None
        assert config.get('camera') is not None
        assert config.get('logging') is not None
    
    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        nonexistent_path = temp_dir / 'nonexistent.yaml'
        with pytest.raises(FileNotFoundError):
            Config.load(str(nonexistent_path))
    
    def test_load_malformed_yaml_raises_error(self, temp_dir):
        """Test that malformed YAML raises YAMLError."""
        malformed_file = temp_dir / 'malformed.yaml'
        with open(malformed_file, 'w') as f:
            f.write("invalid: yaml: content:\n  - broken")
        
        with pytest.raises(yaml.YAMLError):
            Config.load(str(malformed_file))
    
    def test_load_empty_file(self, temp_dir):
        """Test loading an empty YAML file."""
        empty_file = temp_dir / 'empty.yaml'
        empty_file.touch()
        
        config = Config.load(str(empty_file))
        assert config is not None
        assert config.to_dict() == {}


class TestDotNotationAccess:
    """Test dot notation configuration access."""
    
    def test_get_nested_value(self, sample_config_file):
        """Test accessing nested configuration values."""
        config = Config.load(str(sample_config_file))
        path = config.get('models.binary_classifier.path')
        assert path == 'test/model.tflite'
    
    def test_get_top_level_value(self, sample_config_file):
        """Test accessing top-level configuration values."""
        config = Config.load(str(sample_config_file))
        models = config.get('models')
        assert isinstance(models, dict)
        assert 'binary_classifier' in models
    
    def test_get_with_default_value(self, sample_config_file):
        """Test getting nonexistent key returns default value."""
        config = Config.load(str(sample_config_file))
        value = config.get('nonexistent.key', 'default_value')
        assert value == 'default_value'
    
    def test_get_nonexistent_without_default(self, sample_config_file):
        """Test getting nonexistent key without default returns None."""
        config = Config.load(str(sample_config_file))
        value = config.get('nonexistent.key')
        assert value is None
    
    def test_get_deeply_nested_value(self, sample_config_file):
        """Test accessing deeply nested values."""
        config = Config.load(str(sample_config_file))
        width = config.get('models.binary_classifier.input_width')
        assert width == 320


class TestEnvironmentVariableOverrides:
    """Test environment variable override functionality."""
    
    def test_env_var_override_string(self, sample_config_file, monkeypatch):
        """Test environment variable overrides string value."""
        monkeypatch.setenv('CRUSTACEAN_LOGGING_LEVEL', 'DEBUG')
        config = Config.load(str(sample_config_file))
        level = config.get('logging.level')
        assert level == 'DEBUG'
    
    def test_env_var_override_integer(self, sample_config_file, monkeypatch):
        """Test environment variable overrides integer value."""
        monkeypatch.setenv('CRUSTACEAN_CAMERA_WIDTH', '1920')
        config = Config.load(str(sample_config_file))
        width = config.get('camera.width')
        assert width == 1920
        assert isinstance(width, int)
    
    def test_env_var_override_boolean_true(self, sample_config_file, monkeypatch):
        """Test environment variable overrides boolean (true)."""
        monkeypatch.setenv('CRUSTACEAN_LOGGING_CONSOLE', 'false')
        config = Config.load(str(sample_config_file))
        console = config.get('logging.console')
        assert console is False
    
    def test_env_var_override_boolean_false(self, sample_config_file, monkeypatch):
        """Test environment variable overrides boolean (false)."""
        monkeypatch.setenv('CRUSTACEAN_LOGGING_CONSOLE', 'true')
        config = Config.load(str(sample_config_file))
        console = config.get('logging.console')
        assert console is True
    
    def test_env_var_override_float(self, sample_config_file, monkeypatch):
        """Test environment variable overrides float value."""
        monkeypatch.setenv('CRUSTACEAN_MODELS_BINARY_CLASSIFIER_INPUT_WIDTH', '320.5')
        config = Config.load(str(sample_config_file))
        width = config.get('models.binary_classifier.input_width')
        assert width == 320.5
        assert isinstance(width, float)
    
    def test_no_env_var_uses_config_value(self, sample_config_file):
        """Test that without env var, config file value is used."""
        config = Config.load(str(sample_config_file))
        level = config.get('logging.level')
        assert level == 'INFO'


class TestConfigModification:
    """Test configuration modification methods."""
    
    def test_set_value(self, sample_config_file):
        """Test setting a configuration value."""
        config = Config.load(str(sample_config_file))
        config.set('logging.level', 'DEBUG')
        assert config.get('logging.level') == 'DEBUG'
    
    def test_set_nested_value(self, sample_config_file):
        """Test setting a deeply nested value."""
        config = Config.load(str(sample_config_file))
        config.set('new.nested.value', 42)
        assert config.get('new.nested.value') == 42
    
    def test_get_section(self, sample_config_file):
        """Test getting an entire configuration section."""
        config = Config.load(str(sample_config_file))
        camera_section = config.get_section('camera')
        assert isinstance(camera_section, dict)
        assert camera_section['width'] == 1280
        assert camera_section['height'] == 720
    
    def test_get_nonexistent_section(self, sample_config_file):
        """Test getting a nonexistent section returns empty dict."""
        config = Config.load(str(sample_config_file))
        section = config.get_section('nonexistent')
        assert section == {}


class TestConfigSerialization:
    """Test configuration save and serialization."""
    
    def test_to_dict(self, sample_config_file):
        """Test converting config to dictionary."""
        config = Config.load(str(sample_config_file))
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'models' in config_dict
        assert 'logging' in config_dict
    
    def test_save_config(self, sample_config_file, temp_dir):
        """Test saving configuration to file."""
        config = Config.load(str(sample_config_file))
        config.set('logging.level', 'DEBUG')
        
        output_path = temp_dir / 'saved_config.yaml'
        config.save(str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Load saved config and verify
        saved_config = Config.load(str(output_path))
        assert saved_config.get('logging.level') == 'DEBUG'
    
    def test_save_creates_directory(self, sample_config_file, temp_dir):
        """Test that save creates parent directories if needed."""
        config = Config.load(str(sample_config_file))
        
        output_path = temp_dir / 'nested' / 'dir' / 'config.yaml'
        config.save(str(output_path))
        
        assert output_path.exists()


class TestConfigRepresentation:
    """Test string representation methods."""
    
    def test_repr(self, sample_config_file):
        """Test __repr__ method."""
        config = Config.load(str(sample_config_file))
        repr_str = repr(config)
        assert 'Config' in repr_str
        assert 'sections' in repr_str
    
    def test_str(self, sample_config_file):
        """Test __str__ method."""
        config = Config.load(str(sample_config_file))
        str_repr = str(config)
        assert 'Config with sections' in str_repr
        assert 'models' in str_repr
        assert 'logging' in str_repr


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_key(self, sample_config_file):
        """Test accessing with empty key."""
        config = Config.load(str(sample_config_file))
        # Empty key should return the entire config
        result = config.get('')
        assert result is None or isinstance(result, dict)
    
    def test_key_with_spaces(self, sample_config_file):
        """Test that keys with spaces are handled."""
        config = Config.load(str(sample_config_file))
        config.set('key.with spaces', 'value')
        assert config.get('key.with spaces') == 'value'
    
    def test_numeric_values_preserved(self, sample_config_file):
        """Test that numeric values maintain their types."""
        config = Config.load(str(sample_config_file))
        width = config.get('models.binary_classifier.input_width')
        assert isinstance(width, int)
        assert width == 320
