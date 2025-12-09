"""
Pytest configuration and shared fixtures for Crustacean Monitoring System tests.

This module provides common fixtures and configuration for all tests.
"""

import sys
from unittest.mock import MagicMock

# Mock tflite_runtime before any imports that use it
# This allows tests to run on platforms where tflite_runtime is not available
mock_tflite = MagicMock()
sys.modules['tflite_runtime'] = mock_tflite
sys.modules['tflite_runtime.interpreter'] = mock_tflite

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Temporary directory path
        
    Cleanup:
        Removes directory after test completes
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_dict():
    """
    Provide a sample configuration dictionary for testing.
    
    Returns:
        dict: Sample configuration
    """
    return {
        'models': {
            'binary_classifier': {
                'path': 'test/model.tflite',
                'input_width': 320,
                'input_height': 180
            }
        },
        'logging': {
            'level': 'INFO',
            'console': True
        },
        'camera': {
            'width': 1280,
            'height': 720
        }
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """
    Create a temporary YAML config file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        sample_config_dict: Sample configuration fixture
        
    Returns:
        Path: Path to temporary config file
    """
    config_path = temp_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return config_path
