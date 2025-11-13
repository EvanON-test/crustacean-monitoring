"""
Logging setup and configuration for Crustacean Monitoring System.

This module provides centralized logging configuration with support for
console and file output, automatic log rotation, and thread-safe logging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(config=None) -> None:
    """
    Configure logging system based on configuration settings.
    
    Sets up console and/or file handlers with rotating file support,
    configures log format with thread names, and sets log levels.
    
    Args:
        config: Configuration object. If None, uses default settings.
        
    Example:
        >>> from crustacean.utils.config import Config
        >>> config = Config.load()
        >>> setup_logging(config)
    """
    global _logging_configured
    
    # Prevent duplicate configuration
    if _logging_configured:
        return
    
    # Get logging configuration or use defaults
    if config is not None:
        log_level = config.get('logging.level', 'INFO')
        console_enabled = config.get('logging.console', True)
        file_enabled = config.get('logging.file', True)
        log_dir = config.get('output.log_dir', 'logs')
        log_filename = config.get('logging.filename', 'crustacean_monitoring.log')
        max_bytes = config.get('logging.max_bytes', 10485760)  # 10MB
        backup_count = config.get('logging.backup_count', 5)
        log_format = config.get('logging.format', 
                               '%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
        date_format = config.get('logging.date_format', '%Y-%m-%d %H:%M:%S')
    else:
        # Default settings
        log_level = 'INFO'
        console_enabled = True
        file_enabled = True
        log_dir = 'logs'
        log_filename = 'crustacean_monitoring.log'
        max_bytes = 10485760  # 10MB
        backup_count = 5
        log_format = '%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Add console handler if enabled
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if file_enabled:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        log_file = log_path / log_filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Mark as configured
    _logging_configured = True
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized (level: {log_level})")
    if file_enabled:
        logger.info(f"Log file: {log_path / log_filename}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Creates a logger with the specified name. If logging hasn't been
    configured yet, sets up basic configuration.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Ensure logging is configured with defaults if not already done
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)


def reset_logging() -> None:
    """
    Reset logging configuration.
    
    This is primarily useful for testing to allow reconfiguration.
    Removes all handlers and resets the configured flag.
    """
    global _logging_configured
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    _logging_configured = False


def set_log_level(level: str) -> None:
    """
    Change the log level for all handlers.
    
    Args:
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> set_log_level('DEBUG')
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log level changed to {level.upper()}")


def get_log_level() -> str:
    """
    Get the current log level.
    
    Returns:
        Current log level as string
    """
    root_logger = logging.getLogger()
    level = logging.getLevelName(root_logger.level)
    return level
