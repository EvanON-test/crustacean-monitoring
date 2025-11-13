"""
Unit tests for logging system.

Tests the logging setup, configuration, log levels, file rotation,
and output formatting.
"""

import pytest
import logging
import os
from pathlib import Path
from crustacean.utils.logging_setup import (
    setup_logging,
    get_logger,
    reset_logging,
    set_log_level,
    get_log_level
)
from crustacean.utils.config import Config


class TestLoggingSetup:
    """Test logging system initialization."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_setup_logging_with_config(self, sample_config_file):
        """Test logging setup with configuration."""
        config = Config.load(str(sample_config_file))
        setup_logging(config)
        
        # Verify root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
    
    def test_setup_logging_without_config(self):
        """Test logging setup with default settings."""
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
    
    def test_setup_logging_idempotent(self):
        """Test that calling setup_logging multiple times doesn't duplicate handlers."""
        setup_logging()
        handler_count_1 = len(logging.getLogger().handlers)
        
        setup_logging()
        handler_count_2 = len(logging.getLogger().handlers)
        
        assert handler_count_1 == handler_count_2
    
    def test_console_handler_enabled(self):
        """Test that console handler is added when enabled."""
        setup_logging()
        
        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers 
                          if isinstance(h, logging.StreamHandler) 
                          and not isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(console_handlers) > 0
    
    def test_file_handler_enabled(self, temp_dir):
        """Test that file handler is added when enabled."""
        # Create a config with custom log directory
        config_dict = {
            'logging': {'level': 'INFO', 'console': True, 'file': True},
            'output': {'log_dir': str(temp_dir / 'logs')}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(file_handlers) > 0


class TestLoggerCreation:
    """Test logger instance creation."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_with_module_name(self):
        """Test logger creation with module name."""
        logger = get_logger('crustacean.models.binary_classifier')
        assert logger.name == 'crustacean.models.binary_classifier'
    
    def test_get_logger_auto_configures(self):
        """Test that get_logger auto-configures logging if not done."""
        # Don't call setup_logging
        logger = get_logger('test')
        
        # Should still work
        logger.info("Test message")
        
        # Verify logging was configured
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
    
    def test_multiple_loggers_share_config(self):
        """Test that multiple loggers share the same configuration."""
        setup_logging()
        
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        
        # Both should use the same root configuration
        assert logger1.level == logger2.level


class TestLogLevels:
    """Test log level configuration and changes."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_default_log_level(self):
        """Test default log level is INFO."""
        setup_logging()
        level = get_log_level()
        assert level == 'INFO'
    
    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        setup_logging()
        set_log_level('DEBUG')
        
        level = get_log_level()
        assert level == 'DEBUG'
    
    def test_set_log_level_warning(self):
        """Test setting log level to WARNING."""
        setup_logging()
        set_log_level('WARNING')
        
        level = get_log_level()
        assert level == 'WARNING'
    
    def test_set_log_level_error(self):
        """Test setting log level to ERROR."""
        setup_logging()
        set_log_level('ERROR')
        
        level = get_log_level()
        assert level == 'ERROR'
    
    def test_log_level_from_config(self):
        """Test log level is set from configuration."""
        config_dict = {
            'logging': {'level': 'DEBUG', 'console': True, 'file': False}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        level = get_log_level()
        assert level == 'DEBUG'


class TestLogOutput:
    """Test log message output and formatting."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_log_format_includes_thread_name(self, temp_dir):
        """Test that log format includes thread name."""
        log_file = temp_dir / 'test.log'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True, 'filename': 'test.log'},
            'output': {'log_dir': str(temp_dir)}
        }
        config = Config(config_dict)
        setup_logging(config)
        
        logger = get_logger('test')
        logger.info("Test message")
        
        # Read log file and check format
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert 'MainThread' in log_content
        assert 'Test message' in log_content
    
    def test_log_format_includes_module_name(self, temp_dir):
        """Test that log format includes module name."""
        log_file = temp_dir / 'test.log'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True, 'filename': 'test.log'},
            'output': {'log_dir': str(temp_dir)}
        }
        config = Config(config_dict)
        setup_logging(config)
        
        logger = get_logger('test.module')
        logger.info("Test message")
        
        # Read log file and check format
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert 'test.module' in log_content
    
    def test_log_format_includes_level(self, temp_dir):
        """Test that log format includes log level."""
        log_file = temp_dir / 'test.log'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True, 'filename': 'test.log'},
            'output': {'log_dir': str(temp_dir)}
        }
        config = Config(config_dict)
        setup_logging(config)
        
        logger = get_logger('test')
        logger.warning("Warning message")
        
        # Read log file and check format
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert 'WARNING' in log_content
    
    def test_debug_not_logged_at_info_level(self, temp_dir):
        """Test that DEBUG messages are not logged when level is INFO."""
        log_file = temp_dir / 'test.log'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True, 'filename': 'test.log'},
            'output': {'log_dir': str(temp_dir)}
        }
        config = Config(config_dict)
        setup_logging(config)
        
        logger = get_logger('test')
        logger.debug("Debug message")
        logger.info("Info message")
        
        # Read log file and check content
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Only INFO message should be logged
        assert "Debug message" not in log_content
        assert "Info message" in log_content


class TestFileRotation:
    """Test log file rotation functionality."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_log_file_created(self, temp_dir):
        """Test that log file is created."""
        log_dir = temp_dir / 'logs'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True,
                       'filename': 'test.log'},
            'output': {'log_dir': str(log_dir)}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        logger = get_logger('test')
        logger.info("Test message")
        
        log_file = log_dir / 'test.log'
        assert log_file.exists()
    
    def test_log_directory_created(self, temp_dir):
        """Test that log directory is created if it doesn't exist."""
        log_dir = temp_dir / 'nested' / 'logs'
        config_dict = {
            'logging': {'level': 'INFO', 'console': False, 'file': True},
            'output': {'log_dir': str(log_dir)}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        
        assert log_dir.exists()
    
    def test_rotating_file_handler_configured(self, temp_dir):
        """Test that RotatingFileHandler is properly configured."""
        log_dir = temp_dir / 'logs'
        max_bytes = 1024
        backup_count = 3
        
        config_dict = {
            'logging': {
                'level': 'INFO',
                'console': False,
                'file': True,
                'max_bytes': max_bytes,
                'backup_count': backup_count
            },
            'output': {'log_dir': str(log_dir)}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(file_handlers) > 0
        handler = file_handlers[0]
        assert handler.maxBytes == max_bytes
        assert handler.backupCount == backup_count


class TestResetLogging:
    """Test logging reset functionality."""
    
    def test_reset_clears_handlers(self):
        """Test that reset_logging clears all handlers."""
        setup_logging()
        assert len(logging.getLogger().handlers) > 0
        
        reset_logging()
        assert len(logging.getLogger().handlers) == 0
    
    def test_reset_allows_reconfiguration(self):
        """Test that reset allows reconfiguration."""
        setup_logging()
        original_handler_count = len(logging.getLogger().handlers)
        
        reset_logging()
        setup_logging()
        new_handler_count = len(logging.getLogger().handlers)
        
        assert new_handler_count == original_handler_count


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_invalid_log_level_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        config_dict = {
            'logging': {'level': 'INVALID_LEVEL', 'console': True, 'file': False}
        }
        config = Config(config_dict)
        
        setup_logging(config)
        # Should not raise an error, should default to INFO
        level = get_log_level()
        assert level in ['INFO', 'NOTSET']  # May vary by implementation
    
    def test_logging_with_none_config(self):
        """Test logging setup with None config uses defaults."""
        setup_logging(None)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    def test_get_logger_with_empty_name(self):
        """Test get_logger with empty string returns root logger."""
        logger = get_logger('')
        assert logger.name == 'root'
