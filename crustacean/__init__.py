"""
Crustacean Monitoring System

A real-time computer vision pipeline for detecting and analyzing crustaceans
(crabs and lobsters) on edge devices.
"""

__version__ = "2.0.0"
__author__ = "Crustacean Monitoring Team"

# Package-level imports for convenience
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
]
