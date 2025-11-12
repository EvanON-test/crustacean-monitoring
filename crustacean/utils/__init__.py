"""
Shared utility modules.

This package contains utilities for configuration management, logging setup,
custom exceptions, and performance profiling.
"""

from crustacean.utils.config import Config
from crustacean.utils.logging_setup import setup_logging, get_logger
from crustacean.utils.exceptions import (
    CrustaceanError,
    ConfigurationError,
    ModelLoadError,
    ModelNotLoadedError,
    CameraInitError,
    InferenceError,
    ThreadError,
)
from crustacean.utils.profiling import PerformanceProfiler

__all__ = [
    "Config",
    "setup_logging",
    "get_logger",
    "CrustaceanError",
    "ConfigurationError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "CameraInitError",
    "InferenceError",
    "ThreadError",
    "PerformanceProfiler",
]
