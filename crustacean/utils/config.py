"""
Configuration management for Crustacean Monitoring System.

This module provides centralized configuration loading from YAML files
with support for dot notation access, default values, and environment
variable overrides.
"""

import os
import yaml
from typing import Any, Optional, Dict
from pathlib import Path


class Config:
    """
    Configuration manager for the Crustacean Monitoring System.
    
    Loads configuration from YAML files and provides convenient access
    to nested configuration values using dot notation.
    
    Example:
        >>> config = Config.load('config/my_config.yaml')
        >>> model_path = config.get('models.binary_classifier.path')
        >>> threshold = config.get('realtime.motion_detection_threshold', 15)
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config with a configuration dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        """
        self._config = config_dict
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """
        Load configuration from a YAML file.
        
        If no path is provided, attempts to load from:
        1. Environment variable CRUSTACEAN_CONFIG
        2. config/default_config.yaml
        
        If the file doesn't exist, creates it from the default configuration.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            
        Returns:
            Config instance with loaded configuration
            
        Raises:
            FileNotFoundError: If config file not found and cannot create default
            yaml.YAMLError: If YAML file is malformed
        """
        # Determine config file path
        if config_path is None:
            config_path = os.environ.get('CRUSTACEAN_CONFIG', 'config/default_config.yaml')
        
        config_file = Path(config_path)
        
        # If file doesn't exist, try to create from default
        if not config_file.exists():
            default_path = Path('config/default_config.yaml')
            if default_path.exists() and config_path != str(default_path):
                # Copy default to requested location
                import shutil
                config_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(default_path, config_file)
            elif not default_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}\n"
                    f"Default configuration also missing: {default_path}"
                )
        
        # Load YAML file
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
        
        if config_dict is None:
            config_dict = {}
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Supports environment variable overrides using the pattern:
        CRUSTACEAN_<SECTION>_<KEY> (e.g., CRUSTACEAN_LOGGING_LEVEL)
        
        Args:
            key: Configuration key in dot notation (e.g., 'models.binary_classifier.path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('models.binary_classifier.path')
            'processing/binary_classifier/save/DS1_A_200_128.tflite'
            >>> config.get('nonexistent.key', 'default_value')
            'default_value'
        """
        # Check for environment variable override
        env_key = self._key_to_env_var(key)
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Navigate through nested dictionary
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
            
        Example:
            >>> config.set('logging.level', 'DEBUG')
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'models', 'camera')
            
        Returns:
            Dictionary containing section configuration
            
        Example:
            >>> camera_config = config.get_section('camera')
            >>> print(camera_config['width'])
            1280
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def _key_to_env_var(key: str) -> str:
        """
        Convert dot notation key to environment variable name.
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            
        Returns:
            Environment variable name (e.g., 'CRUSTACEAN_LOGGING_LEVEL')
        """
        return 'CRUSTACEAN_' + key.replace('.', '_').upper()
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Attempts to parse as int, float, bool, or returns string.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value with appropriate type
        """
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def __repr__(self) -> str:
        """String representation of Config object."""
        return f"Config({len(self._config)} sections)"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        sections = ', '.join(self._config.keys())
        return f"Config with sections: {sections}"
